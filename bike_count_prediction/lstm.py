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
# steps: the size of the cell?
#RNN_LAYERS = [{'steps': TIMESTEPS}]
# fully connected layer
#DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 3000
# if use data from 2017-10  - 2018- 03 as training data, there are 4000 data samples
# that is 20 batches an epoch. Run 30 epoches, -> 600 steps
BATCH_SIZE = 60
#PRINT_STEPS = TRAINING_STEPS / 100
N_HIDDEN = 30
LEARNING_RATE = 0.001

class generateData(object):
    def __init__(self, rawdata, timesteps, batchsize):
        self.timesteps = timesteps
        self.batchsize = batchsize
        self.rawdata = rawdata
        self.train_batch_id = 0

        X, y = self.load_csvdata()
        self.X = X['train']
        self.y = y['train']

    # https://github.com/tgjeon/TensorFlow-Tutorials-for-Time-Series/blob/master/lstm_predictor.py
    def rnn_data(self, data, labels=False):
        """
        creates new data frame based on previous observation
          * example:
            l = [1, 2, 3, 4, 5]
            time_steps = 2
            -> labels == False [[1, 2], [2, 3], [3, 4]] #Data frame for input with 2 timesteps
            -> labels == True [3, 4, 5] # labels for predicting the next timestep
        """
        rnn_df = []
        for i in range(len(data) - self.timesteps):
            if labels:
                try:
                    rnn_df.append(data.iloc[i + self.timesteps].as_matrix())
                except AttributeError:
                    rnn_df.append(data.iloc[i + self.timesteps])
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
        return dict(train=train_x), dict(train = train_y)


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
#         batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
#                                                   self.batch_size, len(X))])
        self.train_batch_id = min(self.train_batch_id + self.batchsize, len(self.X))
        return batch_data, batch_labels


class SeriesPredictor:
    # input_dim = 1, seq_size = 168, hidden_dim = ?
    def __init__(self, save_path, input_dim, seq_size, hidden_dim):
        # Hyperparameters
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim
        self.save_path = save_path

        # Weight variables and input placeholders
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        # [None, 168, 1] = [batchsize, 168, 1]
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        #self.y = tf.placeholder(tf.float32, [None, seq_size])
        # modified y dimension, only need predict 1 y, not seq_size y
        self.y = tf.placeholder(tf.float32, [None, 1])

        # Cost optimizer
        #self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        #acc_loss = tf.losses.absolute_difference(prediction, self.y, weight)
        self.cost = tf.losses.absolute_difference(self.model(), self.y)
        self.train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)

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
        cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
        # num_examples = batch_size = 100
        #num_examples = tf.shape(self.x)[0]
        # added->  [timesteps, batch_size, 1]
#         _x =  tf.unstack(self.x, num=TIMESTEPS, axis=1)

        # https://stackoverflow.com/questions/44162432/analysis-of-the-output-from-tf-nn-dynamic-rnn-tensorflow-function

        # # tf.Tensor 'rnn_3/transpose:0' shape=(batchsize, 168, 100) dtype=float32
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        # output shape (?, 168, 100)

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * TIMESTEPS + (TIMESTEPS - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, self.hidden_dim]), index)
        out = tf.matmul(outputs, self.W_out) + self.b_out
        # Linear activation, using outputs computed above
        return out


    def train(self, data):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config = config) as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            for i in range(TRAINING_STEPS):
                batch_x, batch_y = data.train_next()
               # batch_test_x, batch_test_y = data.test_next()

                _, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: batch_x, self.y: batch_y})
                if i % 10 == 0:
                    print('step: {}\t\ttrain err: {}'.format(i, train_err))


            #save_path = self.saver.save(sess, 'model.ckpt')
            save_path = self.saver.save(sess, self.save_path +'model.ckpt')
            print('Model saved to {}'.format(save_path))

    def test(self, sess, data):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, self.save_path +'model.ckpt')
        #batch_test_x, batch_test_y = data.test_next()
        output = sess.run(self.model(), feed_dict={self.x: data.X})
        return output


class lstm:
    def __init__(self, train_obj, save_path,
                    TIMESTEPS,
                    TRAINING_STEPS, LEARNING_RATE):
        self.train_obj = train_obj
        self.train_df = train_obj.train_df
        self.test_df = train_obj.test_df
        self.save_path = save_path

        globals()['TIMESTEPS']  = TIMESTEPS
        # globals()['BATCH_SIZE']  = BATCH_SIZE
        globals()['TRAINING_STEPS']  = TRAINING_STEPS
        globals()['LEARNING_RATE']  = LEARNING_RATE
        #globals()['LATENT_CHANNEL'] = self.latent_test_series.shape[-1]

        print('len(self.train_df):', len(self.train_df))
        print('len(self.test_df):', len(self.test_df))



        # get prediction results
        print('get prediction results')
        self.fea = 'total_count'  # total_count of frement bridge west and east
        #print('self.train_df[self.fea]: ', self.train_df[self.fea])

        self.lstm_predicted = self.run_lstm_for_single_grid(self.train_df[self.fea], self.test_df[self.fea])

    # TODO: save results for every grid's predction, and recover after resuming
    # input_series: time series for a single grid
    def run_lstm_for_single_grid(self, train_series, test_series):
        # lstm_predicted = pd.DataFrame(np.nan,
        #                         index=self.test_df[self.train_obj.predict_start_time: self.train_obj.predict_end_time].index,
        #                         columns= [self.fea])
        tf.reset_default_graph()

        predictor = SeriesPredictor(self.save_path, input_dim=1, seq_size=TIMESTEPS, hidden_dim=N_HIDDEN)
        #data = data_loader.load_series('international-airline-passengers.csv')
        data = generateData(train_series, TIMESTEPS, BATCH_SIZE)
        # # DEBUG:
        #print('train data.x', data.X)
        # create batches, feed batches into predictor
        predictor.train(data)
        print('finished training')

        # prep test data
        test_data = generateData(test_series, TIMESTEPS, BATCH_SIZE)
        print('test_data.y: ', test_data.y)

        with tf.Session() as sess:
            predicted_vals = predictor.test(sess, test_data)[:,0]
            print('predicted_vals', np.shape(predicted_vals))

        predicted = np.transpose(predicted_vals)
        # debug
        print('predicted: ', predicted)

        #mse = mean_absolute_error(y['test'], predicted)
        #print ("Error: %f" % mse)
        rmse = np.sqrt((np.asarray((np.subtract(predicted, test_data.y))) ** 2).mean())
        # this previous code for rmse was incorrect, array and not matricies was needed: rmse = np.sqrt(((predicted - y['test']) ** 2).mean())
        #score = mean_squared_error(predicted, test_data.y)
        #nmse = score / np.var(test_data.y) # should be variance of original data and not data from fitted model, worth to double check
        mae = mean_absolute_error(predicted, test_data.y)
        print("RSME: %f" % rmse)
        #print("NSME: %f" % nmse)
        #print("MSE: %f" % score)
        print('MAE: %f' %mae)


        # path = os.path.join(self.save_path, 'lstm_temp/')
        # if not os.path.exists(path):
        #     print("path doesn't exist. trying to make", path)
        #     os.makedirs(path)
        filename = os.path.join(self.save_path, self.fea+'.csv')
        # lstm_predicted[self.fea] = predicted_vals.tolist()
        temp_res = pd.DataFrame({self.fea:predicted_vals.tolist()})
        print('saving files to ', filename)
        temp_res.to_csv(filename)

        txt_path = os.path.join(self.save_path, 'lstm_bikecount_output.txt')

        with open(txt_path, 'w') as the_file:
            the_file.write('rmse for lstm\n')
            the_file.write(str(rmse))
            the_file.write('mae for lstm\n')
            the_file.write(str(mae))
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
