import json
import sys

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd

import glob # read multiple files
import os
import os.path
from os import getcwd
from os.path import join
from os.path import basename   # get file name
import collections
import matplotlib.pyplot as plt
import time

import datetime
from datetime import timedelta
import datetime_utils
import evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


LOG_DIR = './ops_logs'
# use 10 time points to predict next
TIMESTEPS = 168
PREDICTION_STEPS = 6
# steps: the size of the cell?
#RNN_LAYERS = [{'steps': TIMESTEPS}]
# fully connected layer
#DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 3000
# if use data from 2017-10  - 2018- 03 as training data, there are 4000 data samples
# that is 20 batches an epoch. Run 30 epoches, -> 600 steps
BATCH_SIZE = 128
#PRINT_STEPS = TRAINING_STEPS / 100
N_HIDDEN = 128 # previously 30
LEARNING_RATE = 0.001


class generateData(object):
    def __init__(self, rawdata, timesteps, batchsize):
        self.timesteps = timesteps
        self.batchsize = batchsize
        self.rawdata = rawdata
        self.train_batch_id = 0

        X, y, decoder_inputs = self.load_csvdata()
        self.X = X['train']
        self.y = y['train']
        self.decoder_inputs = decoder_inputs['train']

    # https://github.com/tgjeon/TensorFlow-Tutorials-for-Time-Series/blob/master/lstm_predictor.py
    def rnn_data(self, data, labels=False, decoder_inputs = False):
        """
        creates new data frame based on previous observation
          * example:
            l = [0, 1, 2, 3, 4, 5, 6, 7]
            time_steps = 3, prediction_steps = 2
            -> labels == False [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]] #Data frame for input with 2 timesteps
            -> labels == True [[3,4], [4, 5],[5,6] [6, 7]] # labels for predicting the next timesteps
            -> decoder inputs: [[2, 3], [3,4], [4,5], [5,6]]
        """
        rnn_df = []
        for i in range(len(data) - self.timesteps - PREDICTION_STEPS + 1):
            if labels:
                try:
                    # TODO: better not hard code 'total_count' column
                    # rnn_df.append(data['total_count'].iloc[i + self.timesteps].as_matrix())
                    data_ = data['beacon_hill'].iloc[i + self.timesteps:  i + self.timesteps + PREDICTION_STEPS].as_matrix()
                    rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
                except AttributeError:
                    data_ = data['beacon_hill'].iloc[i + self.timesteps:  i + self.timesteps + PREDICTION_STEPS].as_matrix()
                    rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
            elif decoder_inputs:
                data_ = data.iloc[i + self.timesteps-1:  i + self.timesteps + PREDICTION_STEPS-1].as_matrix()
                rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

            else:
                data_ = data.iloc[i: i + self.timesteps].as_matrix()
                rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

        return np.array(rnn_df, dtype=np.float32)


    # load raw data
    def load_csvdata(self):
        data = self.rawdata
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        train_x = self.rnn_data(data)
        train_y =self.rnn_data(data, labels = True)
        train_y = np.squeeze(train_y, axis=2)
        # expand dim to [batchsize, 1]
        # train_y = np.expand_dims(train_y, axis=1)
        # debug
        print('train_x.shape: ', train_x.shape) # (41448, 168, 1)
        print('train_y.shape: ', train_y.shape)

        # decoder input
        train_decoder_inputs = self.rnn_data(data, labels = False, decoder_inputs = True )
        print('train_decoder_inputs.shape: ', train_decoder_inputs.shape)

        return dict(train=train_x), dict(train = train_y), dict(train = train_decoder_inputs)


    # input train_x, train_y or test_x or test_y
    def train_next(self):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.train_batch_id == len(self.X):
            self.train_batch_id = 0
        batch_data = (self.X[self.train_batch_id:min(self.train_batch_id +
                                                  self.batchsize, len(self.X))])
        batch_labels = (self.y[self.train_batch_id:min(self.train_batch_id +
                                                  self.batchsize, len(self.y))])
        batch_decoder_inputs = (self.decoder_inputs[self.train_batch_id:min(self.train_batch_id +
                                                  self.batchsize, len(self.decoder_inputs))])
#         batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
#                                                   self.batch_size, len(X))])
        self.train_batch_id = min(self.train_batch_id + self.batchsize, len(self.X))
        return batch_data, batch_labels, batch_decoder_inputs


class SeriesPredictor:
    # input_dim = 1, seq_size = 168, hidden_dim = ?
    def __init__(self, save_path, input_dim, seq_size, hidden_dim,
                resume_training, checkpoint_path):
        # Hyperparameters
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim
        self.save_path = save_path

        ## DEBUG:
        print('input_dim: ', input_dim)
        self.resume_training = resume_training
        self.checkpoint_path = checkpoint_path

        # Weight variables and input placeholders
        # self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        # self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')

        # encoder inputs
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        # decoder outputs
        self.y = tf.placeholder(tf.float32, [None, PREDICTION_STEPS])
        self.decoder_inputs = tf.placeholder(tf.float32, [None, PREDICTION_STEPS, input_dim])

        self.global_step = tf.Variable(0, trainable=False)

        # Cost optimizer
        self.train_pred, _ = self.model()
        self.cost = tf.losses.absolute_difference(self.train_pred, self.y)

        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                       5000, 0.8, staircase=True)

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost, global_step = self.global_step)
        # Auxiliary ops
        self.saver = tf.train.Saver()

    # T: 168,  batch_size = num_examples, input_size = 1
    # # batch_size sequences of length 10 with 2 values for each timestep
    # input = get_batch(X, batch_size).reshape([batch_size, 10, 2])

    #  ref: https://www.damienpontifex.com/2017/12/06/understanding-tensorflows-rnn-inputs-outputs-and-shapes/
    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        #cell = rnn.BasicLSTMCell(self.hidden_dim)
        with tf.variable_scope('encoding'):
            encoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, name = 'encoder_cell')
            # num_examples = batch_size = 100
            #num_examples = tf.shape(self.x)[0]
            # added->  [timesteps, batch_size, 1]
    #         _x =  tf.unstack(self.x, num=TIMESTEPS, axis=1)

            # https://stackoverflow.com/questions/44162432/analysis-of-the-output-from-tf-nn-dynamic-rnn-tensorflow-function

            # # tf.Tensor 'rnn_3/transpose:0' shape=(batchsize, 168, 100) dtype=float32
            encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder_cell, self.x, dtype=tf.float32)
            # output shape (?, 168, 100)
        with tf.variable_scope('decoding'):
            decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, name = 'decoder_cell')
            # At this point decoder_cell output is a hidden_units sized vector at every timestep
            decoder_outputs, decoder_states = tf.nn.dynamic_rnn(decoder_cell, self.decoder_inputs,
                                initial_state=encoder_states, dtype=tf.float32)
            print('decoder_outputs.shape: ', decoder_outputs.shape) # decoder_outputs.shape:
            out = tf.contrib.layers.fully_connected(decoder_outputs, 1)
            out = tf.squeeze(out, axis = 2)
        print('out.shape: ', out.shape)

        # Hack to build the indexing and retrieve the right output.
        # batch_size = tf.shape(outputs)[0]
        # # Start indices for each sample
        # index = tf.range(0, batch_size) * TIMESTEPS + (TIMESTEPS - 1)
        # # Indexing
        # outputs = tf.gather(tf.reshape(outputs, [-1, self.hidden_dim]), index)
        # out = tf.matmul(outputs, self.W_out) + self.b_out

        # Linear activation, using outputs computed above
        return out, encoder_states

    # def inference(self):
    #     return

    def train(self, data, test_data):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config = config) as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())

            if self.resume_training:
                if self.checkpoint_path is not None:
                    self.saver.restore(sess, self.checkpoint_path)
                else:
                    self.saver.restore(sess, tf.train.latest_checkpoint(self.save_path))
                # check global step
                print("global step: ", sess.run([self.global_step]))
                print("Model restore finished, current globle step: %d" % self.global_step.eval())
                start_epoch = self.global_step.eval()
            else:
                start_epoch = 0

            # training
            loss_per100 = 0
            for i in range(start_epoch,TRAINING_STEPS):
                batch_x, batch_y, batch_decoder_inputs = data.train_next()

                _, train_err = sess.run([self.train_op, self.cost],
                            feed_dict={self.x: batch_x, self.y: batch_y, self.decoder_inputs: batch_decoder_inputs})
                loss_per100 += train_err
                if i % 100 == 0 and i!= 0:
                    # print('step: {}\t\ttrain err: {}'.format(i, train_err))
                    loss_per100 = float(loss_per100/100)
                    print('step: {}\t\ttrain err per100: {}'.format(i, loss_per100))

                    # Testing
                    _, test_err = sess.run([self.train_op, self.cost],
                            feed_dict={self.x: test_data.X, self.y: test_data.y, self.decoder_inputs:  test_data.decoder_inputs})

                    pred = sess.run([self.train_pred],
                                feed_dict={self.x: test_data.X, self.y: test_data.y, self.decoder_inputs:  test_data.decoder_inputs})

                    # save epoch statistics to csv
                    ecoch_res_df = pd.DataFrame([[loss_per100, test_err]],
                        columns=[ 'train_loss', 'test_lost'])
                    print('step: {}\t\ttest err: {}'.format(i, test_err))
                    print('prediction VS GT: ', pred, test_data.y)
                    res_csv_path = self.save_path + 'err_df' +'.csv'
                    with open(res_csv_path, 'a') as f:
                        # Add header if file is being created, otherwise skip it
                        ecoch_res_df.to_csv(f, header=f.tell()==0)
                    loss_per100 = 0

            #save_path = self.saver.save(sess, 'model.ckpt')
            save_path = self.saver.save(sess, self.save_path +'model.ckpt', global_step=self.global_step)
            print('Model saved to {}'.format(save_path))

    def test(self, sess, data):
        tf.get_variable_scope().reuse_variables()
        # self.saver.restore(sess, self.save_path +'model.ckpt', global_step=self.global_step)
        self.saver.restore(sess, tf.train.latest_checkpoint(self.save_path))

        output, _ = sess.run(self.model(), feed_dict={self.x: data.X, self.decoder_inputs: data.decoder_inputs})
        return output


class lstm:
    def __init__(self, train_obj, save_path,
                    TIMESTEPS,
                    TRAINING_STEPS, LEARNING_RATE,
                     is_inference = False, checkpoint_path = None,
                     resume_training = False, train_dir = None):
        self.train_obj = train_obj
        self.train_df = train_obj.train_df
        self.test_df = train_obj.test_df
        self.save_path = save_path

        globals()['TIMESTEPS']  = TIMESTEPS
        # globals()['BATCH_SIZE']  = BATCH_SIZE
        globals()['TRAINING_STEPS']  = TRAINING_STEPS
        globals()['LEARNING_RATE']  = LEARNING_RATE
        # globals()['LATENT_CHANNEL'] = self.latent_test_series.shape[-1]

        print('len(self.train_df):', len(self.train_df))
        print('len(self.test_df):', len(self.test_df))

        self.is_inference = is_inference
        self.checkpoint_path = checkpoint_path

        self.resume_training = resume_training
        self.train_dir = train_dir


        # get prediction results
        print('get prediction results')
        self.fea = 'beacon_hill'  # total_count of frement bridge west and east
        #print('self.train_df[self.fea]: ', self.train_df[self.fea])

        if resume_training == False:
            self.lstm_predicted = self.run_lstm_for_single_grid(self.train_df, self.test_df)
        else:
            self.lstm_predicted = self.run_lstm_for_single_grid(self.train_df, self.test_df,
                                        self.train_dir, self.checkpoint_path)

    # TODO: save results for every grid's predction, and recover after resuming
    # input_series: time series for a single grid
    def run_lstm_for_single_grid(self, train_series, test_series,
            resume_training = False, checkpoint_path = None):
        tf.reset_default_graph()

        predictor = SeriesPredictor(self.save_path, input_dim= len(list(train_series)), seq_size=TIMESTEPS, hidden_dim=N_HIDDEN,
                                resume_training =resume_training , checkpoint_path = checkpoint_path)
        data = generateData(train_series, TIMESTEPS, BATCH_SIZE)
        test_data = generateData(test_series, TIMESTEPS, BATCH_SIZE)
        # # DEBUG:
        #print('train data.x', data.X)
        # create batches, feed batches into predictor
        predictor.train(data, test_data)
        print('finished training')
        # prep test data
        print('test_data.y: ', test_data.y)
        print('test_data.decoder_inputs: ', test_data.decoder_inputs)

        # inference
        with tf.Session() as sess:
            predicted_vals = predictor.test(sess, test_data)
            print('predicted_vals', np.shape(predicted_vals))

        # predicted = np.transpose(predicted_vals)
        # debug
        # print('predicted: ', predicted)
        rmse = np.sqrt((np.asarray((np.subtract(predicted_vals, test_data.y))) ** 2).mean())
        mae = mean_absolute_error(predicted_vals, test_data.y)
        print("RSME: %f" % rmse)
        print('MAE: %f' %mae)

        filename = os.path.join(self.save_path, self.fea+'.csv')
        temp_res = pd.DataFrame({self.fea:predicted_vals.tolist()})
        print('saving files to ', filename)
        temp_res.to_csv(filename)

        txt_path = os.path.join(self.save_path, 'lstm_airquality_output.txt')

        with open(txt_path, 'w') as the_file:
            the_file.write('rmse for lstm\n')
            the_file.write(str(rmse))
            the_file.write('mae for lstm\n')
            the_file.write(str(mae))
            the_file.write('epochs\n')
            the_file.write(str(TRAINING_STEPS))
            the_file.write('batsch size\n')
            the_file.write(str(BATCH_SIZE))
            the_file.write('n_hidden\n')
            the_file.write(str(N_HIDDEN))

            the_file.close()
    #     plot_predicted, = plt.plot(predicted, label='predicted')
    #     plot_test, = plt.plot(test_data.y, label='test')
    #     plt.legend(handles=[plot_predicted, plot_test])
        return temp_res


    def run_lstim_all_grids(self, steps=1):
        # store temporary output (predictions for each grid)
        path = os.path.join(self.train_obj.save_path, 'lstm_temp/')
        if not os.path.exists(path):
            print("path doesn't exist. trying to make", path)
            os.makedirs(path)


        lstm_predicted = pd.DataFrame(np.nan,
                                index=self.test_df[self.train_obj.predict_start_time: self.train_obj.predict_end_time].index,
                                columns=list(self.test_df))
        print('shape of lstm_predicted:', len(lstm_predicted))
        print('self.train_obj.predict_start_time: ', self.train_obj.predict_start_time)
        print('self.train_obj.predict_end_time: ', self.train_obj.predict_end_time)
        index = 0
        for fea in list(self.test_df):
            print('creating estimation for: ', fea)
            print('index of fea: ', index)
            #pred = self.run_lstm_for_single_grid(self.train_df[fea], self.test_df[fea])
            #lstm_predicted[fea] = pred.tolist()
            # save prediction result
            # filename = path + fea+'.csv'
            filename = os.path.join(path, fea+'.csv')

            print('saved file name: ', filename)
            if os.path.exists(filename):
                print('prediction exists, read from file for ', fea)
                pred_list = pd.read_csv(filename, index_col = 0)
                pred = np.array(pred_list[fea].tolist())
            else:
                pred = self.run_lstm_for_single_grid(self.train_df[fea], self.test_df[fea])
                lstm_predicted[fea] = pred.tolist()
                temp_res = pd.DataFrame({fea:pred.tolist()})
                print('saving files to ', filename)
                temp_res.to_csv(filename)

            count = 0
            for dt in datetime_utils.datetime_range(self.train_obj.test_start_time, self.train_obj.actual_end_time, {'hours':steps}):
                dt_str = pd.to_datetime(datetime_utils.datetime_to_str(dt))
                    # note:  dt_end_str = dt_str+ 24*7 = '2018-04-08 00:00:00' -> it actually the time we want to predict
                dt_end_str =pd.to_datetime(datetime_utils.datetime_to_str(dt+self.train_obj.window -  self.train_obj.step))
                input_series = self.test_df[fea][dt_str:dt_end_str]
                    # predicted_timestamp = '2018-04-08 01:00:00 '
                predicted_timestamp = dt+self.train_obj.window
                predicted_timestamp_str = pd.to_datetime(datetime_utils.datetime_to_str(predicted_timestamp))

                lstm_predicted.loc[predicted_timestamp_str, fea] = pred[count]
                count+=1
            index+=1
        return lstm_predicted
