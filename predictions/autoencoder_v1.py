# v1: vanilla autoencoder
#  train autoencoder for urban features
# first, stack features to form a 3D tensor [H, W, time, channels]
# train autoencoder to learn laten representation as [H, W, 1, 1]
# the latent represetnation is learned from a sequece of 168 hours
# last updated: October, 2019


import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import math
import datetime
from datetime import timedelta
import datetime_utils
# from loss import mean_diff
# from loss import equal_mean
# from loss import multi_var_mean_diff
# from loss import multi_var_fine_grained_diff
# from loss import pairwise_fairloss
# from loss import weighted_MAE_loss
# from loss import differential_weighed_MAE_loss
# from loss import binary_loss
import os

import tensorflow.python.keras
import tensorflow.contrib.keras as keras
from tensorflow.python.keras import backend as K


HEIGHT = 32
WIDTH = 20
TIMESTEPS = 168
# without exogenous data, the only channel is the # of trip starts
BIKE_CHANNEL = 1
NUM_2D_FEA = 4 # slope = 2, bikelane = 2
NUM_1D_FEA = 3  # temp/slp/prec

CHANNEL = 27


BATCH_SIZE = 32
# actually epochs
TRAINING_STEPS = 50
# TRAINING_STEPS = 1
LEARNING_RATE = 0.003


def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)



class Autoencoder:
    # input_dim = 1, seq_size = 168,
    def __init__(self, intersect_pos_set,
                    demo_mask_arr, channel, time_steps, height, width):

        self.time_steps = time_steps
        self.width = width
        self.height = height
        self.channel = channel  # 27

        self.x = tf.placeholder(tf.float32, (None, time_steps, height,width,channel), name="input")
        self.y = tf.placeholder(tf.float32, (None, time_steps, height,width,channel), name="target")

        # this is usefor Batch normalization.
        # https://towardsdatascience.com/pitfalls-of-batch-norm-in-tensorflow-and-sanity-checks-for-training-networks-e86c207548c8
        self.is_training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, trainable=False)

        '''
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                       5000, 0.96, staircase=True)
        '''

    '''
    inputs_: feature tensor: input shape: [None, timestep, height, width, channels]
             e.g. [None, 168, 32, 20, 9]
    to get the latent representation, obtain: encoded [None, 1, 32, 20, 1]
    '''
    def vanilla_autoencoder(self, inputs_):
        padding = 'SAME'
        stride = [1,1,1]
        with tf.name_scope("encoding"):
            # [168, 32, 20, 9]
            conv1 = tf.layers.conv3d(inputs= inputs_, filters=16, kernel_size=(3,3,3), padding= padding, strides = stride, activation=my_leaky_relu)
            # now [168, 32, 20, 16]
            maxpool1 = tf.layers.max_pooling3d(conv1, pool_size=(2,1,1), strides=(2,1,1), padding= padding)
            # [84, 32, 20, 16]
            conv2 = tf.layers.conv3d(inputs=maxpool1, filters=32, kernel_size=(3,3,3), padding= padding, strides = stride, activation=my_leaky_relu)
            # [84, 32, 20, 32]
            maxpool2 = tf.layers.max_pooling3d(conv2, pool_size=(3,1,1), strides=(3,1,1), padding= padding)
            # [28, 32, 20, 32]
            conv3 = tf.layers.conv3d(inputs=maxpool2, filters=32, kernel_size=(3,3,3), padding= padding , strides = stride, activation=my_leaky_relu)
            # [28, 32, 20, 32]
            maxpool3 = tf.layers.max_pooling3d(conv3, pool_size=(4,1,1), strides=(4,1,1), padding= padding)
            # [7, 32, 20, 32]

            conv4 = tf.layers.conv3d(inputs=maxpool3, filters=32, kernel_size=(3,3,3), padding= padding , strides = stride, activation=my_leaky_relu)
            # [7, 32, 20, 32]
            maxpool4 = tf.layers.max_pooling3d(conv4, pool_size=(7,1,1), strides=(7,1,1), padding= padding)
            # [1, 32, 20, 32]

            encoded = tf.layers.conv3d(inputs=maxpool4, filters=1, kernel_size=[3,3,3], padding='same', activation=my_leaky_relu)
            # [1, 32, 20, 1]
            print('encoded.shape', encoded)

        with tf.name_scope("decoding"):
                        # decoder
            # [1, 32, 20, 1]-> [168, 32, 20, 9]
            deconv1 = tf.layers.conv3d_transpose(inputs=encoded, filters=32, kernel_size=(3,3,3), padding= padding , strides = stride, activation=my_leaky_relu)
            # [1, 32, 20, 32]
            # https://www.tensorflow.org/api_docs/python/tf/keras/backend/resize_volumes
            unpool1 = K.resize_volumes(deconv1,7,1,1,"channels_last")
            # [7, 32, 20, 32]

            # upsample1 = tf.image.resize_images(encoded, size=(7,7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # # [7, 32, 20, 32]
            deconv2 = tf.layers.conv3d_transpose(inputs=unpool1, filters=32, kernel_size=(3,3,3), padding= padding , strides = stride, activation=my_leaky_relu)
            # # [28, 32, 20, 32]
            unpool2 = K.resize_volumes(deconv2,4,1,1,"channels_last")
            # # [28, 32, 20, 32]
            deconv2 = tf.layers.conv3d_transpose(inputs=unpool2, filters=32, kernel_size=(3,3,3), padding= padding , strides = stride, activation=my_leaky_relu)
            # # [28, 32, 20, 32]
            unpool3 = K.resize_volumes(deconv2,3,1,1,"channels_last")
            # # [84, 32, 20, 32]
            deconv3 = tf.layers.conv3d_transpose(inputs=unpool3, filters=32, kernel_size=(3,3,3), padding= padding , strides = stride, activation=my_leaky_relu)
            # now # # [16, 20, 64, 64] = [units, time step, width, height]  -> [20, 64, 64, 1]
            unpool4 = K.resize_volumes(deconv3,2,1,1,"channels_last")
            # # [168, 32, 20, 32]

            # [168, 32, 20, 9]
            output = tf.layers.conv3d(inputs=unpool4, filters=self.channel, kernel_size=[3,3,3], padding='same', activation=my_leaky_relu)
            # [none, 1, 20, 64, 64] -> [none, 20, 64, 64, 1]
            # 0 , 1 ,2 ,3, 4 ->  0, 2,3,4,1
            # output = tf.transpose(output, perm=[0, 2,3,4,1])

            # (?, 168, 32, 20, 9)
            print('output.shape: ', output.shape)

        return output, encoded


    def generate_fixlen_timeseries(self, rawdata_arr, timestep = 168):
        raw_seq_list = list()
            # arr_shape: [# of timestamps, w, h]
        arr_shape = rawdata_arr.shape
        for i in range(0, arr_shape[0] - (timestep)+1):
            start = i
            end = i+ (timestep )
                # temp_seq = rawdata_arr[start: end, :, :]
            temp_seq = rawdata_arr[start: end]
            raw_seq_list.append(temp_seq)
        raw_seq_arr = np.array(raw_seq_list)
        raw_seq_arr = np.swapaxes(raw_seq_arr,0,1)
        return raw_seq_arr



    def create_mini_batch(self, start_idx, end_idx, data_1d, data_2d, data_3d):
        # data_3d : (45984, 32, 20, 3)
        # data_1d: (45984, 4)
        # data_2d: (32, 20, 20)
        test_size = end_idx - start_idx
        test_data_1d = data_1d[start_idx:end_idx + 168 - 1,:]
        test_data_1d_seq = self.generate_fixlen_timeseries(test_data_1d)
    #     print(test_data_1d_seq.shape)
        test_data_1d_seq = np.expand_dims(test_data_1d_seq, axis=2)
        test_data_1d_seq = np.expand_dims(test_data_1d_seq, axis=3)
        test_data_1d_seq = np.tile(test_data_1d_seq,(1,1, 32,20,1))

        test_data_3d = data_3d[start_idx :end_idx + 168 - 1, :, :]
        test_data_3d_seq = self.generate_fixlen_timeseries(test_data_3d)

        test_data_2d = np.expand_dims(data_2d, axis=0)
        test_data_2d = np.expand_dims(test_data_2d, axis=0)
        test_data_2d = np.tile(test_data_2d,(168, test_data_1d_seq.shape[1],1,1,1))

        # print(test_data_1d_seq.shape)
    #     print(test_data_2d.shape)
    #     print(test_data_3d_seq.shape)
        concat_list = [test_data_1d_seq, test_data_2d, test_data_3d_seq]
        test_x = np.concatenate(concat_list, axis=4)
        # input should be (None, time_steps, height,width,channel)
        test_x = np.swapaxes(test_x,0,1)
        # print(test_x.shape)
        return test_x



    def train_autoencoder(self, data_1d, data_2d, data_3d, train_hours,
                     demo_mask_arr, save_folder_path,
                       epochs=1, batch_size=32):
        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                       5000, 0.96, staircase=True)
        reconstructed, encoded = self.vanilla_autoencoder(self.x)

        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
        # [1, 32, 20, 1]  -> [1, 1, 32, 20, 1]
        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr_expanded, 0)
        # [1, 32, 20, 1] -> [batchsize, 32, 20, 1]
        # batchsize = tf.shape(prediction)[0]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(reconstructed)[0],TIMESTEPS,1,1, CHANNEL])

        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)
        acc_loss = tf.losses.absolute_difference(reconstructed, self.y, weight)
        cost = acc_loss

        with tf.name_scope("training"):
        # with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = self.global_step)


        saver = tf.train.Saver()
        test_result = list()

        if not os.path.exists(save_folder_path):
            os.makedirs(save_path)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start_time = datetime.datetime.now()
            # temporary
            # train_hours = 200
            # train_hours: train_start_time = '2014-02-01',train_end_time = '2018-10-31',
            if train_hours%batch_size ==0:
                iterations = int(train_hours/batch_size)
            else:
                iterations = int(train_hours/batch_size) + 1

            for epoch in range(epochs):
                print('Epoch', epoch, 'started', end='')
                epoch_loss = 0
                final_output = list()
                            # mini batch
                for itr in range(iterations):
                    start_idx = itr*batch_size
                    if train_hours < (itr+1)*batch_size:
                        end_idx = train_hours
                    else:
                        end_idx = (itr+1)*batch_size
                    print('start_idx, end_idx', start_idx, end_idx)
                    mini_batch_x = self.create_mini_batch(start_idx, end_idx, data_1d, data_2d, data_3d)

                    batch_cost, _ = sess.run([cost, optimizer], feed_dict={self.x: mini_batch_x,
                                                                    self.y: mini_batch_x})
                    # get encoded representation
                    # # [None, 1, 32, 20, 1]
                    batch_output = sess.run([encoded], feed_dict={self.x: mini_batch_x,
                                                                    self.y: mini_batch_x})
                    final_output.extend(batch_output)

                    epoch_loss += batch_cost
                    if itr%10 == 0:
                        print("Epoch: {}/{}...".format(itr, epoch),
                            "Training loss: {:.4f}".format(batch_cost))

                # report loss per epoch
                epoch_loss = epoch_loss/ iterations
                print('epoch: ', epoch, 'Trainig Set Epoch total Cost: ',epoch_loss)
                save_path = saver.save(sess, save_folder_path +'autoencoder_' +str(epoch)+'.ckpt', global_step=self.global_step)
                # save_path = saver.save(sess, './autoencoder.ckpt')
                print('Model saved to {}'.format(save_path))

                # save epoch statistics to csv
                ecoch_res_df = pd.DataFrame([[epoch_loss]],
                    columns=[ 'train_loss'])

                res_csv_path = save_folder_path + 'autoencoder_ecoch_res_df' +'.csv'

                with open(res_csv_path, 'a') as f:
                    # Add header if file is being created, otherwise skip it
                    ecoch_res_df.to_csv(f, header=f.tell()==0)

                # save results to txt
                txt_name = save_folder_path + 'AE_df_' +  '.txt'
                with open(txt_name, 'w') as the_file:
                    #the_file.write('Only account for grids that intersect with city boundary \n')
                    the_file.write('epoch\n')
                    the_file.write(str(epoch)+'\n')
                    # the_file.write('lamda\n')
                    # the_file.write(str(lamda) + '\n')
                    the_file.write(' epoch_loss:\n')
                    the_file.write(str(epoch_loss) + '\n')
                    the_file.write('\n')
                    the_file.close()

                # plot results
                print('saving train_test plots')
                train_test = pd.read_csv(save_folder_path  + 'autoencoder_ecoch_res_df' +'.csv')
                # train_test = train_test.loc[:, ~train_test.columns.str.contains('^Unnamed')]
                train_test[['train_loss']].plot()
                plt.savefig(save_folder_path + 'total_loss_inprogress.png')
                # train_test[['train_acc', 'test_acc']].plot()
                # plt.savefig(save_folder_path + 'acc_loss_inprogress.png')
                # train_test[['train_fair', 'test_fair']].plot()
                # plt.savefig(save_folder_path + 'fair_loss_inprogress.png')
                plt.close()

                if epoch == epochs-1:
                    final_output = np.array(final_output)
                    test_result.extend(final_output)

            # encoded_res = np.array(test_result)
            encoded_res = test_result
            output_arr = encoded_res[0]
            for i in range(1,len(encoded_res)):
                output_arr = np.concatenate((output_arr, encoded_res[i]), axis=0)

        # This is the latent representation (9337, 1, 32, 20, 1)
        return output_arr




    def train_autoencoder_from_checkpoint(self, weather_seq_arr, crime_seq_arr, data_2d,
                    lamda, demo_mask_arr, save_folder_path, checkpoint_path,
                      keep_rate=0.7, epochs=10, batch_size=64):
        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                       5000, 0.96, staircase=True)
        reconstructed, encoded = self.vanilla_autoencoder(self.x)

        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
        # [1, 32, 20, 1]  -> [1, 1, 32, 20, 1]
        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr_expanded, 0)
        # [1, 32, 20, 1] -> [batchsize, 32, 20, 1]
        # batchsize = tf.shape(prediction)[0]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(reconstructed)[0],TIMESTEPS,1,1, CHANNEL])

        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)
        acc_loss = tf.losses.absolute_difference(reconstructed, self.y, weight)
        cost = acc_loss

        with tf.name_scope("training"):
        # with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = self.global_step)


        saver = tf.train.Saver()
        test_result = list()

        if not os.path.exists(save_folder_path):
            os.makedirs(save_path)

        # keep results for plotting
        # train_acc_loss = []
        # test_acc_loss = []

        # ecoch_res_df['train_acc'] = train_acc_loss
        # ecoch_res_df['test_acc'] = test_acc_loss


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start_time = datetime.datetime.now()

            if checkpoint_path is not None:
                saver.restore(sess, checkpoint_path)
            else:
                saver.restore(sess, tf.train.latest_checkpoint(save_folder_path))
            # check global step
            print("global step: ", sess.run([self.global_step]))
            print("Model restore finished, current globle step: %d" % self.global_step.eval())

            # get new epoch num
            print("int(len(x_train_data) / batch_size +1): ", int(len(crime_seq_arr) / batch_size +1))
            start_epoch_num = tf.div(self.global_step, int(len(crime_seq_arr) / batch_size +1))
            #self.global_step/ (len(x_train_data) / batch_size +1) -1
            print("start_epoch_num: ", start_epoch_num.eval())
            start_epoch = start_epoch_num.eval()

            if len(crime_seq_arr)%batch_size ==0:
                iterations = int(len(crime_seq_arr)/batch_size)
            else:
                iterations = int(len(crime_seq_arr)/batch_size) + 1
                        # run epochs
                        # global step = epoch * len(x_train_data) + itr
            for epoch in range(epochs):
                print('Epoch', epoch, 'started', end='')
                epoch_loss = 0
                final_output = list()
                            # mini batch
                for itr in range(iterations):
                    # (None, 168, 3)
                    mini_batch_weather = weather_seq_arr[itr*batch_size: (itr+1)*batch_size]
                    # (None,  168, 1,3)
                    mini_batch_weather = np.expand_dims(mini_batch_weather, axis=2)
                    # (None,  168, 1,1,3)
                    mini_batch_weather = np.expand_dims(mini_batch_weather, axis=3)
                    # (None, 168, 32,20, 3)
                    mini_batch_weather = np.tile(mini_batch_weather,(1,1, 32,20,1))
                    # (None, 168, 32, 20)
                    mini_batch_crime = crime_seq_arr[itr*batch_size: (itr+1)*batch_size]
                    # None, 168, 32, 20, 1
                    mini_batch_crime = np.expand_dims(mini_batch_crime, axis=4)
                    # (32, 20, 5) => None, 168, 32, 20, 5
                    # (1, 32, 20, 5)
                    mini_batch_data_2d = np.expand_dims(data_2d, axis=0)
                    # # (1, 1, 32, 20, 5)
                    mini_batch_data_2d = np.expand_dims(mini_batch_data_2d, axis=0)
                    mini_batch_data_2d = np.tile(mini_batch_data_2d,(mini_batch_weather.shape[0],TIMESTEPS, 1,1,1))

                    # print('mini_batch_weather.shape: ', mini_batch_weather.shape)
                    # print('mini_batch_crime.shape: ', mini_batch_crime.shape)

                    mini_batch_x = np.concatenate([mini_batch_weather,mini_batch_crime], axis=4)
                    mini_batch_x = np.concatenate([mini_batch_x,mini_batch_data_2d], axis=4)
            #         print('mini_batch_x.shape: ', mini_batch_x.shape)
            #         mini_batch_y = y_train_data[itr*batch_size: (itr+1)*batch_size]
                    batch_cost, _ = sess.run([cost, optimizer], feed_dict={self.x: mini_batch_x,
                                                                    self.y: mini_batch_x})
                    # get encoded representation
                    # # [None, 1, 32, 20, 1]
                    batch_output = sess.run([encoded], feed_dict={self.x: mini_batch_x,
                                                                    self.y: mini_batch_x})
                    final_output.extend(batch_output)

                    epoch_loss += batch_cost
                    if itr%10 == 0:
                        print("Epoch: {}/{}...".format(itr, epoch),
                            "Training loss: {:.4f}".format(batch_cost))

                # report loss per epoch
                epoch_loss = epoch_loss/ iterations
                print('epoch: ', epoch, 'Trainig Set Epoch total Cost: ',epoch_loss)
                save_path = saver.save(sess, save_folder_path +'autoencoder_v1_' +'_'+str(epoch)+'.ckpt', global_step=self.global_step)
                # save_path = saver.save(sess, './autoencoder.ckpt')
                print('Model saved to {}'.format(save_path))


         # save epoch statistics to csv
                ecoch_res_df = pd.DataFrame([[epoch_loss]],
                    columns=[ 'train_loss'])

                res_csv_path = save_folder_path + 'autoencoder_v1_ecoch_res_df_' +'.csv'

                with open(res_csv_path, 'a') as f:
                    # Add header if file is being created, otherwise skip it
                    ecoch_res_df.to_csv(f, header=f.tell()==0)

                # save results to txt
                txt_name = save_folder_path + 'AE_df_' +  '.txt'
                with open(txt_name, 'w') as the_file:
                    #the_file.write('Only account for grids that intersect with city boundary \n')
                    the_file.write('epoch\n')
                    the_file.write(str(epoch)+'\n')
                    # the_file.write('lamda\n')
                    # the_file.write(str(lamda) + '\n')
                    the_file.write(' epoch_loss:\n')
                    the_file.write(str(epoch_loss) + '\n')
                    # the_file.write('Testing Set Fair Cost\n')
                    # the_file.write(str(test_fair_loss/itrs)+ '\n')
                    # the_file.write('Testing Set Accuracy Cost\n')
                    # the_file.write(str(test_acc_loss/itrs)+ '\n')
                    the_file.write('\n')
                    the_file.close()


                # plot results
                print('saving train_test plots')
                train_test = pd.read_csv(save_folder_path  + 'autoencoder_ecoch_res_df_' +'.csv')
                # train_test = train_test.loc[:, ~train_test.columns.str.contains('^Unnamed')]
                train_test[['train_loss']].plot()
                plt.savefig(save_folder_path + 'total_loss_inprogress.png')
                # train_test[['train_acc', 'test_acc']].plot()
                # plt.savefig(save_folder_path + 'acc_loss_inprogress.png')
                # train_test[['train_fair', 'test_fair']].plot()
                # plt.savefig(save_folder_path + 'fair_loss_inprogress.png')
                plt.close()

                if epoch == epochs-1:
                    final_output = np.array(final_output)
                    test_result.extend(final_output)

            # encoded_res = np.array(test_result)
            encoded_res = test_result
            output_arr = encoded_res[0]
            for i in range(1,len(encoded_res)):
                output_arr = np.concatenate((output_arr, encoded_res[i]), axis=0)

        # This is the latent representation (9337, 1, 32, 20, 1)
        return output_arr


    def inference_autoencoder(self, data_1d, data_2d, data_3d, train_hours,
                     demo_mask_arr, save_folder_path,checkpoint_path,
                       epochs=1, batch_size=32):
        # starter_learning_rate = LEARNING_RATE
        # learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
        #                                5000, 0.96, staircase=True)
        reconstructed, encoded = self.vanilla_autoencoder(self.x)

        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
        # [1, 32, 20, 1]  -> [1, 1, 32, 20, 1]
        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr_expanded, 0)
        # [1, 32, 20, 1] -> [batchsize, 32, 20, 1]
        # batchsize = tf.shape(prediction)[0]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(reconstructed)[0],TIMESTEPS,1,1, CHANNEL])

        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)
        acc_loss = tf.losses.absolute_difference(reconstructed, self.y, weight)
        cost = acc_loss

        saver = tf.train.Saver()
        test_result = list()
        test_cost = 0
        final_output = list()

        if not os.path.exists(save_folder_path):
            os.makedirs(save_path)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint_path)
            print("global step: ", sess.run([self.global_step]))
            print("Model restore finished, current globle step: %d" % self.global_step.eval())

            start_time = datetime.datetime.now()

            if train_hours%batch_size ==0:
                iterations = int(train_hours/batch_size)
            else:
                iterations = int(train_hours/batch_size) + 1

                            # mini batch
            for itr in range(iterations):

                start_idx = itr*batch_size
                if train_hours < (itr+1)*batch_size:
                    end_idx = train_hours
                else:
                    end_idx = (itr+1)*batch_size
                print('start_idx, end_idx', start_idx, end_idx)
                mini_batch_x = self.create_mini_batch(start_idx, end_idx, data_1d, data_2d, data_3d)

                batch_cost = sess.run([cost], feed_dict={self.x: mini_batch_x,
                                                                    self.y: mini_batch_x})
                    # get encoded representation
                    # # [None, 1, 32, 20, 1]
                batch_output = sess.run([encoded], feed_dict={self.x: mini_batch_x,
                                                                    self.y: mini_batch_x})
                final_output.extend(batch_output)
                test_cost += batch_output

                # epoch_loss += batch_cost
                if itr%10 == 0:
                        print("Epoch: {}...".format(itr),
                            "Training loss: {:.4f}".format(batch_cost))

                # report loss per epoch
            test_cost = test_cost/ iterations
            print('Trainig Set total Cost: ',test_cost)

            final_output = np.array(final_output)
            test_result.extend(final_output)

            # encoded_res = np.array(test_result)
            encoded_res = test_result
            output_arr = encoded_res[0]
            for i in range(1,len(encoded_res)):
                output_arr = np.concatenate((output_arr, encoded_res[i]), axis=0)

        # This is the latent representation (9337, 1, 32, 20, 1)
        return output_arr




'''
fixed lenght time window: 168 hours
'''
class Autoencoder_entry:
    def __init__(self, train_obj, data_1d, data_2d, data_3d, intersect_pos_set,
                    demo_mask_arr, save_path,
                    HEIGHT, WIDTH, TIMESTEPS, CHANNEL, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                     is_inference = False, checkpoint_path = None,
                     resume_training = False, train_dir = None
                     ):
                     #  if s_inference = True, do inference only
        self.train_obj = train_obj
        self.train_hours = train_obj.train_hours
        self.data_1d = data_1d
        self.data_2d = data_2d
        self.data_3d = data_3d

        self.intersect_pos_set = intersect_pos_set
        self.demo_mask_arr = demo_mask_arr
        self.save_path = save_path

        globals()['HEIGHT']  = HEIGHT
        globals()['WIDTH']  = WIDTH
        globals()['TIMESTEPS']  = TIMESTEPS
        globals()['CHANNEL']  = CHANNEL
        globals()['BATCH_SIZE']  = BATCH_SIZE
        globals()['TRAINING_STEPS']  = TRAINING_STEPS
        globals()['LEARNING_RATE']  = LEARNING_RATE

        print('Conv3D recieved: ')
        print('HEIGHT: ', HEIGHT)
        print('start learning rate: ',LEARNING_RATE)

        self.is_inference = is_inference
        self.checkpoint_path = checkpoint_path
        self.resume_training = resume_training
        self.train_dir = train_dir

        # ignore non-intersection cells in test_df
        # this is for evaluation
        # self.test_df_cut = self.test_df.loc[:,self.test_df.columns.isin(list(self.intersect_pos_set))]

        if is_inference == False:
            if resume_training == False:
                    # get prediction results
                    print('training from scratch, and get prediction results')
                    # predicted_vals: (552, 30, 30, 1)
                    self.latent_representation = self.run_autoencoder()
                    np.save(self.save_path +'latent_representation.npy', self.latent_representation)
            else:
                    # resume training
                    print("resume training, and get prediction results")
                    self.latent_representation  = self.run_resume_training()
                    np.save(self.save_path +'resumed_prediction_arr.npy', self.latent_representation)
        else:
            # inference only
            print('get inference results')
            self.latent_representation  = self.run_inference_autoencoder()
            np.save(self.save_path +'autoencoder_inference_arr.npy', self.latent_representation)


        # calculate performance using only cells that intersect with city boundary
        # do evaluation using matrix format
        # self.evaluation()

        # convert predicted_vals to pandas dataframe with timestamps
        # self.conv3d_predicted = self.arr_to_df()


    def run_autoencoder(self):
        tf.reset_default_graph()
        # self, channel, time_steps, height, width
        predictor = Autoencoder(self.intersect_pos_set,
                     self.demo_mask_arr, channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        # (9337, 1, 32, 20, 1)
        latent_representation = predictor.train_autoencoder(
                        self.data_1d, self.data_2d, self.data_3d, self.train_hours,
                         self.demo_mask_arr, self.save_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return latent_representation




    # run training and testing together
    def run_resume_training(self):
        tf.reset_default_graph()
        # self, channel, time_steps, height, width
        predictor = Conv3DPredictor(self.intersect_pos_set, self.demo_sensitive, self.demo_pop,
                                    self.pop_g1, self.pop_g2,self.grid_g1, self.grid_g2, self.fairloss,
                                    # self.data_2d, self.data_1d.X, self.data_2d, self.data_1d_test.X,
                                     self.lamda, self.demo_mask_arr, channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH,
                                    )
        #data = data_loader.load_series('international-airline-passengers.csv')
        # rawdata, timesteps, batchsize
        self.train_data = generateData(self.train_arr, TIMESTEPS, BATCH_SIZE)
        # train_x shape should be : [num_examples (batch size), time_step (168), feature_dim (1)]
        # prep test data, split test_arr into test_x and test_y
        self.test_data = generateData(self.test_arr, TIMESTEPS, BATCH_SIZE)
        print('test_data.y.shape', self.test_data.y.shape)

        if self.train_arr_1d is not None:
            self.train_data_1d = generateData_1d(self.train_arr_1d, TIMESTEPS, BATCH_SIZE)
            self.test_data_1d = generateData_1d(self.test_arr_1d, TIMESTEPS, BATCH_SIZE)
            print('test_data_1d.y.shape', self.test_data_1d.y.shape)
            predicted_vals = predictor.train_from_checkpoint(self.train_data.X, self.train_data.y,
                        self.test_data.X, self.test_data.y,
                        self.demo_sensitive, self.demo_pop, self.pop_g1, self.pop_g2,
                        self.grid_g1, self.grid_g2, self.fairloss,
                        self.lamda, self.demo_mask_arr,
                        self.data_2d, self.train_data_1d.X, self.data_2d, self.test_data_1d.X,
                          self.train_dir,self.beta, self.checkpoint_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)
        else:
            print('No 1d feature')
            self.train_data_1d = None
            self.test_data_1d = None
            predicted_vals = predictor.train_from_checkpoint(self.train_data.X, self.train_data.y,
                        self.test_data.X, self.test_data.y,
                        self.demo_sensitive, self.demo_pop, self.pop_g1, self.pop_g2,
                        self.grid_g1, self.grid_g2, self.fairloss,
                        self.lamda, self.demo_mask_arr,
                        self.data_2d, None, self.data_2d, None,
                          self.train_dir,self.beta,self.checkpoint_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        predicted = predicted_vals.flatten()
        y = self.test_data.y.flatten()
        #mse = mean_absolute_error(y['test'], predicted)
        #print ("Error: %f" % mse)
        rmse = np.sqrt((np.asarray((np.subtract(predicted, y))) ** 2).mean())
        mae = mean_absolute_error(predicted, y)
        print('Metrics for all grids: ')
        print("RSME: %f" % rmse)
        print('MAE: %f' %mae)
        return predicted_vals



    # run inference only
    def run_inference_autoencoder(self):
        tf.reset_default_graph()
        predictor = Autoencoder(self.intersect_pos_set,
                     self.demo_mask_arr, channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        # (9337, 1, 32, 20, 1)
        latent_representation = predictor.inference_autoencoder(
                        self.data_1d, self.data_2d, self.data_3d, self.train_hours,
                         self.demo_mask_arr, self.save_path, self.checkpoint_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return latent_representation




    # evaluate rmse and mae with grids that intersect with city boundary
    def evaluation(self):
        sample_pred_squeeze = np.squeeze(self.predicted_vals)
        test_squeeze = np.squeeze(self.test_data.y)
        pred_shape = self.predicted_vals.shape
        mse = 0
        mae = 0
        count = 0
        for i in range(0, pred_shape[0]):
            temp_image = sample_pred_squeeze[i]
            test_image = test_squeeze[i]
            # rotate
            temp_rot = np.rot90(temp_image, axes=(1,0))
            test_rot= np.rot90(test_image, axes=(1,0))
            for c in range(pred_shape[1]):
                for r in range(pred_shape[2]):
                    temp_str = str(r)+'_'+str(c)

                    if temp_str in self.intersect_pos_set:
                        #print('temp_str: ', temp_str)
                        count +=1
                        mse += (test_rot[r][c] - temp_rot[r][c]) ** 2
                        mae += abs(test_rot[r][c] - temp_rot[r][c])

        rmse = math.sqrt(mse / (pred_shape[0] * len(self.intersect_pos_set)))
        mae = mae / (pred_shape[0] * len(self.intersect_pos_set))
        '''
        BATCH_SIZE = 32
        # actually epochs
        TRAINING_STEPS = 150
        LEARNING_RATE = 0.005
        '''
        print('BATCH_SIZE: ', BATCH_SIZE)
        print('TRAINING_STEPS: ', TRAINING_STEPS)
        print('LEARNING_RATE: ',LEARNING_RATE)
        print('rmse and mae with grids that intersect with city boundary')
        print('rmse: ', rmse)
        print('mae ', mae)
        print('count ', count)

    # convert predicted result tensor back to pandas dataframe
    def arr_to_df(self):
        convlstm_predicted = pd.DataFrame(np.nan,
                                index=self.test_df_cut[self.train_obj.predict_start_time: self.train_obj.predict_end_time].index,
                                columns=list(self.test_df_cut))

        sample_pred_squeeze = np.squeeze(self.predicted_vals)
        # test_squeeze = np.squeeze(self.test_data.y)
        pred_shape = self.predicted_vals.shape

        # loop through time stamps
        for i in range(0, pred_shape[0]):
            temp_image = sample_pred_squeeze[i]
            # test_image = test_squeeze[i]
            # rotate
            temp_rot = np.rot90(temp_image, axes=(1,0))
        #     test_rot= np.rot90(test_image, axes=(1,0))

            dt = datetime_utils.str_to_datetime(self.train_obj.test_start_time) + datetime.timedelta(hours=i)
            # dt_str = pd.to_datetime(datetime_utils.datetime_to_str(dt))
            predicted_timestamp = dt+self.train_obj.window
            predicted_timestamp_str = pd.to_datetime(datetime_utils.datetime_to_str(predicted_timestamp))
            # print('predicted_timestamp_str: ', predicted_timestamp_str)

            for c in range(pred_shape[1]):
                for r in range(pred_shape[2]):
                    temp_str = str(r)+'_'+str(c)

                    if temp_str in self.intersect_pos_set:
                        #print('temp_str: ', temp_str)
                        # count +=1

                        convlstm_predicted.loc[predicted_timestamp_str, temp_str] = temp_rot[r][c]
        return convlstm_predicted
