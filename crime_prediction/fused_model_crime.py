# fusion model with fairness metric2 (individual-based)
# fixed length window: e.g. 168 hours
# added data augmentation

# v2: optional data agumentation and permutation, usually turned off

# TODO: add 3d 911 calls data


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
import os


# image size subjects to change
# may use 32 * 32 to train, and crop out central areas
# since there may be effects of padded cells when doing convolution.
HEIGHT = 32
WIDTH = 20
TIMESTEPS = 56
# without exogenous data, the only channel is the # of trip starts
BIKE_CHANNEL = 2
NUM_2D_FEA = 15 # slope = 2, bikelane = 2
NUM_1D_FEA = 3  # temp/slp/prec


BATCH_SIZE = 32
# actually epochs
TRAINING_STEPS = 200
#TRAINING_STEPS = 10

LEARNING_RATE = 0.003

#save_folder_path = './fairness_res/'

def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)


class generateData_3d_feature(object):
    def __init__(self, input_data, timesteps, batchsize):
        self.timesteps = timesteps
        self.batchsize = batchsize
        self.rawdata = input_data
        self.train_batch_id = 0

        X, y = self.load_data()
        # x should be [batchsize, time_steps, height, width,channel]
        self.X = X['train']
        # y should be [batchsize, height, width, channel]
        self.y = y['train']


    # load raw data
    # raw_data.shape for mnist: (20, 10000, 64, 64)
    def load_data(self):
        data = self.rawdata
        train_x = data[:self.timesteps, :, :, :, :]
        train_y = data[self.timesteps:,:, :, :, 0]

        # reshape x to [None, time_steps, height, width,channel]
        # train_x = np.expand_dims(train_x, axis=4)
        # transpose
        train_x = np.swapaxes(train_x,0,1)
        #sample_train_x = np.reshape(sample_train_x, [-1, time_steps, height, width, 1])
        # transpose y to [batch_size, height, width, channel]
        #sample_train_y = np.reshape(sample_train_y, [-1,  height, width, 1])
        train_y = np.expand_dims(train_y, axis=4)
        # transpose
        train_y = np.swapaxes(train_y,0,1)
        # sqeeze to [batch_size, height, width, channel]
        train_y = np.squeeze(train_y, axis = 1)

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

        self.train_batch_id = min(self.train_batch_id + self.batchsize, len(self.X))
        return batch_data, batch_labels



class generateData(object):
    def __init__(self, input_data, timesteps, batchsize):
        self.timesteps = timesteps
        self.batchsize = batchsize
        self.rawdata = input_data
        self.train_batch_id = 0

        X, y = self.load_data()
        # x should be [batchsize, time_steps, height, width,channel]
        self.X = X['train']
        # y should be [batchsize, height, width, channel]
        self.y = y['train']


    # load raw data
    # raw_data.shape for mnist: (20, 10000, 64, 64)
    def load_data(self):
        data = self.rawdata
        train_x = data[:self.timesteps, :, :, :]
        train_y = data[self.timesteps:,:, :, :]

        # reshape x to [None, time_steps, height, width,channel]
        train_x = np.expand_dims(train_x, axis=4)
        # transpose
        train_x = np.swapaxes(train_x,0,1)
        #sample_train_x = np.reshape(sample_train_x, [-1, time_steps, height, width, 1])
        # transpose y to [batch_size, height, width, channel]
        #sample_train_y = np.reshape(sample_train_y, [-1,  height, width, 1])
        train_y = np.expand_dims(train_y, axis=4)
        # transpose
        train_y = np.swapaxes(train_y,0,1)
        # sqeeze to [batch_size, height, width, channel]
        train_y = np.squeeze(train_y, axis = 1)

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

        self.train_batch_id = min(self.train_batch_id + self.batchsize, len(self.X))
        return batch_data, batch_labels

class generateData_1d(object):
    def __init__(self, input_data, timesteps, batchsize):
        self.timesteps = timesteps
        self.batchsize = batchsize
        self.rawdata = input_data
        self.train_batch_id = 0
        #self.test_batch_id = 0

        X, y = self.load_data()
        # x should be [batchsize, time_steps, height, width,channel]
        self.X = X['train']
        # y should be [batchsize, height, width, channel]
        self.y = y['train']

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
    def load_data(self):
        # (169, 1296, 1, 1, 3)
        data = self.rawdata
        train_x = data[:self.timesteps, :, :]
        train_y = data[self.timesteps:,:, :]

        # reshape x to [None, time_steps, height, width,channel]
#         train_x = np.expand_dims(train_x, axis=4)
        # transpose
        train_x = np.swapaxes(train_x,0,1)
        # transpose
        train_y = np.swapaxes(train_y,0,1)
        # sqeeze to [batch_size, height, width, channel]
        train_y = np.squeeze(train_y, axis = 1)

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




class Conv3DPredictor:
    # input_dim = 1, seq_size = 168,
    def __init__(self, intersect_pos_set,
                    # demo_sensitive, demo_pop, pop_g1, pop_g2,
                    #             grid_g1, grid_g2, fairloss,
                                     demo_mask_arr, channel, time_steps, height, width):

        self.time_steps = time_steps
        self.width = width
        self.height = height
        self.channel = channel

        # [batchsize, depth, height, width, channel]
        self.x = tf.placeholder(tf.float32, shape=[None,time_steps, height, width, channel], name = 'x_input')
        #
        self.y = tf.placeholder(tf.float32, shape= [None, height, width, 1], name = 'y_input')
        # [batchsize, 32, 20, 4]
        self.input_2d_feature = tf.placeholder(tf.float32, shape=[None, height, width, NUM_2D_FEA], name = "input_2d_feature")
        # (168, 9336,  3)
        self.input_1d_feature =  tf.placeholder(tf.float32, shape=[None,time_steps, NUM_1D_FEA], name = "input_1d_feature")

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

    # for 3d cnn
    def cnn_model(self, x_train_data, is_training, keep_rate=0.7, seed=None):
    # output from 3d cnn (?, 168, 32, 20, 1)  ->  (?, 32, 20, 1)

        with tf.name_scope("3d_layer_a"):
            '''
            # conv => 16*16*16
            conv1 = tf.layers.conv3d(inputs=x_train_data, filters=16, kernel_size=[3,3,3], padding='same', activation=my_leaky_relu)
            # conv => 16*16*16
            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=my_leaky_relu)
            # pool => 8*8*8
            #pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
            conv3 = tf.layers.conv3d(inputs=conv2, filters=1, kernel_size=[3,3,3], padding='same', activation=my_leaky_relu)
            '''
            conv1 = tf.layers.conv3d(inputs=x_train_data, filters=16, kernel_size=[3,3,3], padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            # conv => 16*16*16
            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            # shape : (?, 168, 32, 20, 16)  -> (?, 1, 32, 20, 16)
            # (pool_depth, pool_height, pool_width)
            # pool2 = tf.layers.average_pooling3d(inputs=conv2, pool_size=[168, 1, 1], strides=1)

            #pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
            # 3d pooling  see https://github.com/tqvinhcs/C3D-tensorflow/blob/master/m_c3d.py


            # the original design, kernel_size = [1,1,1]
            conv3 = tf.layers.conv3d(inputs=conv2, filters=1, kernel_size=[3,3,3], padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)


        # with tf.name_scope("3d_shape_transpose"):
            # cnn3d_bn = tf.layers.batch_normalization(inputs=conv3, training=is_training)

            # transfer (?, 168, 32, 20, 1) to (?,  32, 20, 168)
            # squeeze -> (?, 168, 32, 20)
        cnn3d_bn_squeeze = tf.squeeze(conv3, axis = 4)
            # print('cnn3d_bn_squeeze shape', cnn3d_bn_squeeze.shape)
            # swap axes -> (?, 32, 20, 168) -> [0, 1, 2, 3] -> []
        cnn3d_bn_squeeze = tf.transpose(cnn3d_bn_squeeze, perm=[0,2,3, 1])
            # (?, 168, 32, 20, 1)


        '''
        # transform (?, 1, 32, 20, 16) -> (?, 32, 20, 16)
        # sqeeze (?, 1, 32, 20, 16) -> (?, 32, 20, 16)
        cnn3d_bn_squeeze = tf.squeeze(pool2, axis = 1)
        '''

        with tf.name_scope("3d_layer_b"):
            conv5 = tf.layers.conv2d(
                inputs=cnn3d_bn_squeeze,
                filters=1,
                kernel_size=[1, 1],
                padding="same",
                activation=my_leaky_relu
                #reuse = tf.AUTO_REUSE

            )

        # caveat: Batch Norm usage in TF
        # https://towardsdatascience.com/pitfalls-of-batch-norm-in-tensorflow-and-sanity-checks-for-training-networks-e86c207548c8
        # https://stackoverflow.com/questions/43234667/tf-layers-batch-normalization-large-test-error
        # add BN before fusion
        '''
        Notes on BN:
        - NB should be between conv layers and activation
        - Generally speakiing, when training, BN will use batch mean and variance. When doing inference,
          BN use population mean and variance which is calculated by exponential moving average.
        - However, in time series prediction, batches are feeded in the order of time, therefore,
          different batches are bound to have different mean/variance. Using population mean/var at
          testing time may not be reasonable as mean/var over say 2 months does not neccessarily
          reflect the statistics of the last week. So the task is different from other whose samples
          are assumed to be independent from each other. Keeping "training" == True at both training and
          testing time for time series task may be a better choice.
        '''

        with tf.name_scope("3d_batch_norm_b"):
            conv5_bn = tf.layers.batch_normalization(inputs=conv5, training= is_training)

        out = conv5_bn
        # output size should be [None, height, width, channel]
        return out



    '''
    input: 2d feature tensor: height * width * # of features (batchsize, 32, 20, 4)
    output: (batchsize, 32, 20, 1)

    '''
    def cnn_2d_model(self, x_2d_train_data, is_training, seed=None):
        if x_2d_train_data is None:
            return None


        with tf.name_scope("2d_layer_a"):
            '''
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(x_2d_train_data, 16, 3, padding='same',activation=my_leaky_relu)
            #  Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 16, 3, padding='same',activation=my_leaky_relu)
    #         conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
            '''

             # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(x_2d_train_data, 16, 3, padding='same',activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            #  Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 16, 3, padding='same',activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)
        # with tf.name_scope("2d_batch_norm_a"):

        #     cnn2d_bn = tf.layers.batch_normalization(inputs=conv2, training=is_training)
        #     print('cnn2d_bn shape: ',cnn2d_bn.shape)

        # output should be (?, 32, 20, 1)
        with tf.name_scope("2d_layer_b"):
            conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=1,
                    kernel_size=[1, 1],
                    padding="same",
                    activation=my_leaky_relu
                    #reuse = tf.AUTO_REUSE
                )


        with tf.name_scope("2d_batch_norm_b"):
            conv3_bn = tf.layers.batch_normalization(inputs=conv3, training=is_training)
        out = conv3_bn

        # out = x_2d_train_data
        # output size should be [height, width, 1]
        return out



    '''
    input: 1d feature tensor: height * width * # of features
                (batchsize, # of timestamp, channel), e.g., (32, 168,  3)
    output: (batchsize, 1)

    '''
    # (batchsize, 168, # of features)
    def cnn_1d_model(self, x_1d_train_data, is_training,seed=None):
        if x_1d_train_data is None:
            return None
        with tf.name_scope("1d_layer_a"):
            '''
            # https://www.tensorflow.org/api_docs/python/tf/layers/conv1d
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv1d(x_1d_train_data, 16, 3, padding='same',activation=my_leaky_relu)
            #  Convolution Layer with 64 filters and a kernel size of 3
            # output shape: None, 168,16
            conv2 = tf.layers.conv1d(conv1, 16, 3,padding='same', activation=my_leaky_relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            # Average Pooling   None, 168,16  -> None, 1, 16
            '''
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv1d(x_1d_train_data, 16, 3, padding='same',activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            #  Convolution Layer with 64 filters and a kernel size of 3
            # output shape: None, 168,16
            conv2 = tf.layers.conv1d(conv1, 16, 3,padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)


            # original, use temporal pooling
            conv2 = tf.layers.average_pooling1d( conv2, TIMESTEPS, 1, padding='valid')
            # https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling2d
    #         conv2 = tf.layers.max_pooling2d(conv2, 2, 2)


            '''
            # try 1 filter instead of average pooling
            # shape from None, 168,16 -> None, 168, 1
            conv3 = tf.layers.conv1d(conv2, 1, 3,padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

            # None, 168, 1 ->  None, 1, 168
            conv3 = tf.transpose(conv3, perm=[0,2,1])
            '''


        # None, 1, 168 -> None, 1, 1
        with tf.name_scope("1d_layer_b"):
            conv4 = tf.layers.conv1d(
                    inputs=conv2,
                    filters=1,
                    kernel_size=1,
                    padding="same",
                    activation=my_leaky_relu
                    #reuse = tf.AUTO_REUSE
                )

        with tf.name_scope("1d_batch_norm_b"):
            conv4_bn = tf.layers.batch_normalization(inputs=conv4, training=is_training)

        # squeeze  None, 1, 1  -> None, 1
        conv4_squeeze = tf.squeeze(conv4_bn, axis = 1)
        out = conv4_squeeze
        print('model 1d cnn output :',out.shape )
        # output size should be [None, 1],
        return out



    # prediction_3d: batchsize, 32,20,1
    # prediction_2d: batchsize, 32, 20,1
    # prediction_1d: batchsize, 1
    # output : batchsize, 32,20,1
    def model_fusion(self, prediction_3d, prediction_2d, prediction_1d, is_training):

        # Fuse features using concatenation

        if prediction_2d is None and prediction_1d is None:
            # only prediction_3d has valid prediction
            return prediction_3d
        elif prediction_1d is None:
            # has prediction_3d and prediction_2d
            # fuse_feature: [batchsize, 32, 20, 2]
            fuse_feature = tf.concat([prediction_3d, prediction_2d], 3)
        elif prediction_2d is None:
            # has prediction_3d and prediction_1d
            # fuse_feature: [batchsize, 32, 20, 2]
            # prediction_1d: batchsize, 1  -> duplicate to batch size, 32, 20, 1
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            print('model_fusion prediction_1d: ', prediction_1d.shape)

            prediction_1d_expand = tf.tile(prediction_1d, [1,tf.shape(prediction_3d)[1],
                                                        tf.shape(prediction_3d)[2] ,1])
            print('prediction_1d_expand shape', prediction_1d_expand.shape)
            # batch size, 32, 20, 2
            fuse_feature = tf.concat([prediction_3d, prediction_1d_expand], 3)
        else:
            # used 1d, 2d, and 3d features. Fuse alltogether
            # prediction_2d:  32, 20,1  ->duplicate to  batchsize, 32, 20, 1
        #     prediction_2d_expand = tf.expand_dims(prediction_2d, 0)
        #     prediction_2d_expand = tf.tile(prediction_2d_expand, [tf.shape(prediction_3d)[0], 1,1,1])

            # prediction_1d: batchsize, 1  -> duplicate to batch size, 32, 20, 1
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            print('model_fusion prediction_1d: ', prediction_1d.shape)

            prediction_1d_expand = tf.tile(prediction_1d, [1,tf.shape(prediction_3d)[1],
                                                        tf.shape(prediction_3d)[2] ,1])
            print('prediction_1d_expand shape', prediction_1d_expand.shape)

            fuse_feature = tf.concat([prediction_3d, prediction_2d], 3)
            # batch size, 32, 20, 3
            fuse_feature = tf.concat([fuse_feature, prediction_1d_expand], 3)



        # fuse features using elementwise product
        '''
        if prediction_2d is None and prediction_1d is None:
            # only prediction_3d has valid prediction
            return prediction_3d
        elif prediction_1d is None:
            # has prediction_3d and prediction_2d
            # fuse_feature: [batchsize, 32, 20, 1]
            fuse_feature = tf.multiply(prediction_3d, prediction_2d)
        elif prediction_2d is None:
            # has prediction_3d and prediction_1d
            # fuse_feature: [batchsize, 32, 20, 1]
            # prediction_1d: batchsize, 1  -> duplicate to batch size, 32, 20, 1
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            print('model_fusion prediction_1d: ', prediction_1d.shape)

            prediction_1d_expand = tf.tile(prediction_1d, [1,tf.shape(prediction_3d)[1],
                                                        tf.shape(prediction_3d)[2] ,1])
            print('prediction_1d_expand shape', prediction_1d_expand.shape)
            # batch size, 32, 20, 2
            fuse_feature = tf.multiply(prediction_3d, prediction_1d_expand)
        else:
            # used 1d, 2d, and 3d features. Fuse alltogether
            # prediction_2d:  32, 20,1  ->duplicate to  batchsize, 32, 20, 1
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            print('model_fusion prediction_1d: ', prediction_1d.shape)

            prediction_1d_expand = tf.tile(prediction_1d, [1,tf.shape(prediction_3d)[1],
                                                        tf.shape(prediction_3d)[2] ,1])
            print('prediction_1d_expand shape', prediction_1d_expand.shape)
            # batch size, 32, 20, 1
            fuse_feature = tf.multiply(prediction_3d, prediction_1d_expand)
            fuse_feature = tf.multiply(fuse_feature, prediction_2d)

        '''

        with tf.name_scope("fusion_layer_a"):
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(fuse_feature, 16, 3, padding='same',activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            #  Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 32, 3, padding='same',activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

        # with tf.name_scope("fusion_batch_norm"):
        #     cnn2d_bn = tf.layers.batch_normalization(inputs=conv2, training=is_training)
        #     # (?, 168, 32, 20, 1)
        #     print('cnn2d_bn shape: ',cnn2d_bn.shape)


        # output should be (?, 32, 20, 1)
        with tf.name_scope("fusion_layer_b"):
            conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=1,
                    kernel_size=[1, 1],
                    padding="same",
                    activation=my_leaky_relu
                    #reuse = tf.AUTO_REUSE
                )
        #
        out = conv3
        # output size should be [batchsize, height, width, 1]
        return out



    # data_2d_train, data_1d_train, data_2d_test, data_1d_test: these could be None
    # TODO: fix fairloss for Austin
    def train_neural_network(self, x_train_data, y_train_data, x_test_data, y_test_data,
                     # demo_sensitive, demo_pop,pop_g1, pop_g2,
                     # grid_g1, grid_g2, fairloss_func,
                     demo_mask_arr,
                      data_2d_train, data_1d_train, data_2d_test, data_1d_test,
                      save_folder_path,
                      # beta = math.e,
                      keep_rate=0.7, epochs=10, batch_size=64):

        #global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                       5000, 0.96, staircase=True)

        #prediction = self.cnn_model(self.x, keep_rate, seed=1)
        # fusion model
        prediction_3d = self.cnn_model(self.x, self.is_training, keep_rate, seed=1)

        if data_2d_train is None:
            prediction_2d = None
            # 32,20,1
            #prediction_2d = self.cnn_2d_model(self.input_2d_feature)
        else:
            prediction_2d = self.cnn_2d_model(self.input_2d_feature, self.is_training, )


        if data_1d_train is None:
            # batchsize, 1,1,1
            prediction_1d = None
            #prediction_1d = self.cnn_1d_model(self.input_1d_feature)
        else:
            prediction_1d = self.cnn_1d_model(self.input_1d_feature, self.is_training, )
            #prediction_1d = None

        # fusion
        prediction = self.model_fusion(prediction_3d, prediction_2d, prediction_1d, self.is_training)


        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
        # [1, 32, 20, 1] -> [batchsize, 32, 20, 1]
        # batchsize = tf.shape(prediction)[0]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(prediction)[0],1,1,1])
        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)
        acc_loss = tf.losses.absolute_difference(prediction, self.y, weight)

        cost = acc_loss

        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_input))
        #cost = tf.losses.absolute_difference(prediction, self.y, weight)

        # for batch normalization is_training
        # # see https://github.com/tensorflow/tensorflow/issues/16455
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.name_scope("training"):
        # with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = self.global_step)


        saver = tf.train.Saver()
    #     correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        # iterations = int(len(x_train_data)/batch_size) + 1
        test_result = list()

        if not os.path.exists(save_folder_path):
            os.makedirs(save_path)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())


            start_time = datetime.datetime.now()
            # iterations = int(len(x_train_data)/batch_size) + 1
            if len(x_train_data)%batch_size ==0:
                iterations = int(len(x_train_data)/batch_size)
            else:
                iterations = int(len(x_train_data)/batch_size) + 1
            # run epochs
            # global step = epoch * len(x_train_data) + itr
            for epoch in range(epochs):
                start_time_epoch = datetime.datetime.now()
                print('Epoch', epoch, 'started', end='')
                epoch_loss = 0
                epoch_fairloss = 0
                epoch_accloss = 0
                # mini batch
                for itr in range(iterations):
                    mini_batch_x = x_train_data[itr*batch_size: (itr+1)*batch_size]
                    mini_batch_y = y_train_data[itr*batch_size: (itr+1)*batch_size]
                    # model fusion
                    if data_1d_train is not None:
                        mini_batch_data_1d = data_1d_train[itr*batch_size: (itr+1)*batch_size]
                    else:
                        mini_batch_data_1d = None
                    if data_2d_train is not None:
                        mini_batch_data_2d = np.expand_dims(data_2d_train, 0)
                        mini_batch_data_2d = np.tile(mini_batch_data_2d, [mini_batch_x.shape[0], 1,1,1])
                    else:
                        mini_batch_data_2d = None

                    # 1d, 2d, and 3d
                    if data_1d_train is not None and data_2d_train is not None:

                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.input_1d_feature:mini_batch_data_1d,  self.input_2d_feature: mini_batch_data_2d,
                                                            self.is_training: True   })
                    elif data_1d_train is not None:  # 1d and 3d
                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.input_1d_feature:mini_batch_data_1d,
                                                            self.is_training: True   })
                    elif data_2d_train is not None:
                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.input_2d_feature: mini_batch_data_2d,
                                                            self.is_training: True   })
                    else: # only 3d
                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.is_training: True   })

                    epoch_loss += _cost
                    epoch_accloss += _acc_loss

                    if itr % 10 == 0:
                        #print('epoch: {}, step: {}\t\ttrain err: {}'.format(epoch, itr, _cost))
                        print('epoch: {}, step: {}, train err: {}, mae:{}'.format(epoch, itr, _cost, _acc_loss))

                # report loss per epoch
                epoch_loss = epoch_loss/ iterations
                epoch_accloss = epoch_accloss / iterations

                # train_acc_loss.append(epoch_accloss)
                print('epoch: ', epoch, 'Trainig Set Epoch total Cost: ',epoch_loss)
                print('epoch: ', epoch, 'Trainig Set Epoch accuracy Cost: ',epoch_accloss)

                #  using mini batch in case not enough memory
                test_cost = 0
                test_acc_loss = 0

                final_output = list()

                print('testing')
                start_time_epoch = datetime.datetime.now()
                itrs = int(len(x_test_data)/batch_size) + 1
                for itr in range(itrs):
                    mini_batch_x_test = x_test_data[itr*batch_size: (itr+1)*batch_size]
                    mini_batch_y_test = y_test_data[itr*batch_size: (itr+1)*batch_size]
                    # model fusion
                    if data_1d_test is not None:
                        mini_batch_data_1d_test = data_1d_test[itr*batch_size: (itr+1)*batch_size]
                    else:
                        mini_batch_data_1d_test  = None

                    if data_2d_test is not None:
                        mini_batch_data_2d_test = np.expand_dims(data_2d_test, 0)
                        mini_batch_data_2d_test = np.tile(mini_batch_data_2d_test, [mini_batch_x_test.shape[0], 1,1,1])
                    else:
                        mini_batch_data_2d_test = None

                    if data_1d_test is not None and data_2d_test is not None:
                        #acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test ,
                                            self.is_training: True  })
                        # test_fair_loss += sess.run(fair_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                        #                     self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                        #                     self.is_training: True})
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True})

                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                                        self.is_training: True})
                    elif data_1d_test is not None:
                        #acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,
                                            self.is_training: True  })
                        # test_fair_loss += sess.run(fair_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                        #                     self.input_1d_feature:mini_batch_data_1d_test,
                        #                     self.is_training: True})
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,
                                            self.is_training: True})

                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_1d_feature:mini_batch_data_1d_test,
                                        self.is_training: True})
                    elif data_2d_test is not None:
                                                #acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True  })
                        # test_fair_loss += sess.run(fair_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                        #                     self.input_2d_feature: mini_batch_data_2d_test,
                        #                     self.is_training: True})
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True})

                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_2d_feature: mini_batch_data_2d_test,
                                        self.is_training: True})
                    else:
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                          self.is_training: True  })
                        # test_fair_loss += sess.run(fair_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                        #                     self.is_training: True})
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.is_training: True})
                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.is_training: True})

                    final_output.extend(batch_output)


    #             output = np.array(final_output)

                end_time_epoch = datetime.datetime.now()
                test_time_per_epoch = end_time_epoch - start_time_epoch
                test_time_per_sample = test_time_per_epoch/ len(x_test_data)
                #print(' Testing Set Accuracy:',test_cost/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))
                print(' Testing Set Cost:',test_cost/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))
                # print(' Testing Set Fair Cost:',test_fair_loss/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))
                print(' Testing Set Accuracy Cost:',test_acc_loss/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))
                # test_acc_loss.append(test_acc_loss/itrs)


                #save_folder_path  = './fusion_model_'+ str(lamda)+'/'
                # save globel step for resuming training later
                save_path = saver.save(sess, save_folder_path +'fusion_model_' +'_'+str(epoch)+'.ckpt', global_step=self.global_step)
                print('Model saved to {}'.format(save_path))

                # save epoch statistics to csv
                ecoch_res_df = pd.DataFrame([[epoch_loss, test_cost/itrs, epoch_accloss, test_acc_loss/itrs]],
                    columns=[ 'train_loss','test_loss', 'train_acc', 'test_acc'])

                res_csv_path = save_folder_path + 'ecoch_res_df_' +'.csv'

                with open(res_csv_path, 'a') as f:
                    # Add header if file is being created, otherwise skip it
                    ecoch_res_df.to_csv(f, header=f.tell()==0)

                # save results to txt
                txt_name = save_folder_path + 'fusion_df_' +  '.txt'
                with open(txt_name, 'w') as the_file:
                    #the_file.write('Only account for grids that intersect with city boundary \n')
                    the_file.write('epoch\n')
                    the_file.write(str(epoch)+'\n')
                    the_file.write(' Testing Set Cost:\n')
                    the_file.write(str(test_cost/itrs) + '\n')
                    the_file.write('Testing Set Accuracy Cost\n')
                    the_file.write(str(test_acc_loss/itrs)+ '\n')
                    the_file.write(str(test_acc_loss/itrs)+ '\n')
                    the_file.write('total time of last test epoch\n')
                    the_file.write(str(test_time_per_epoch) + '\n')
                    the_file.write('time per sample\n')
                    the_file.write(str(test_time_per_sample) + '\n')
                    the_file.write('\n')
                    the_file.close()

                if epoch == epochs-1:
                    test_result.extend(final_output)


                # plot results
                print('saving train_test plots')
                train_test = pd.read_csv(save_folder_path  + 'ecoch_res_df_' +'.csv')
                # train_test = train_test.loc[:, ~train_test.columns.str.contains('^Unnamed')]
                train_test[['train_loss', 'test_loss']].plot()
                plt.savefig(save_folder_path + 'total_loss_inprogress.png')
                train_test[['train_acc', 'test_acc']].plot()
                plt.savefig(save_folder_path + 'acc_loss_inprogress.png')
                # train_test[['train_fair', 'test_fair']].plot()
                # plt.savefig(save_folder_path + 'fair_loss_inprogress.png')
                plt.close()

            output = np.array(test_result)

            return output


    # data_2d_train, data_1d_train, data_2d_test, data_1d_test: these could be None
    # resume training from checkpoint.
    # TODO: add switching cities and 1d/2d feature use
    def train_from_checkpoint(self, x_train_data, y_train_data, x_test_data, y_test_data,
                     demo_sensitive, demo_pop,pop_g1, pop_g2,
                     grid_g1, grid_g2, fairloss_func,
                     lamda, demo_mask_arr,
                      data_2d_train, data_1d_train, data_2d_test, data_1d_test,
                      save_folder_path, beta, checkpoint_path,
                      keep_rate=0.7, epochs=10, batch_size=64):

        #global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                       5000, 0.96, staircase=True)

        #prediction = self.cnn_model(self.x, keep_rate, seed=1)
        # fusion model
        prediction_3d = self.cnn_model(self.x, self.is_training, keep_rate, seed=1)

        if data_2d_train is None:
            prediction_2d = None
            # 32,20,1
            #prediction_2d = self.cnn_2d_model(self.input_2d_feature)
        else:
            prediction_2d = self.cnn_2d_model(self.input_2d_feature, self.is_training, )


        if data_1d_train is None:
            # batchsize, 1,1,1
            prediction_1d = None
            #prediction_1d = self.cnn_1d_model(self.input_1d_feature)
        else:
            prediction_1d = self.cnn_1d_model(self.input_1d_feature, self.is_training, )
            #prediction_1d = None

        # fusion
        prediction = self.model_fusion(prediction_3d, prediction_2d, prediction_1d, self.is_training, )

        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
        # [1, 32, 20, 1] -> [batchsize, 32, 20, 1]
        # batchsize = tf.shape(prediction)[0]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(prediction)[0],1,1,1])
        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)
        #fair_loss = group_fairloss(prediction, y_input, demo_sensitive, demo_pop)
        #fair_loss = mean_diff(prediction, self.y, demo_sensitive, demo_pop, pop_g1, pop_g2, demo_mask_arr)
        # multi_var fairness loss: input multi_demo_sensitive shape : [32, 20, 3]
        #fair_loss = multi_var_mean_diff(prediction, self.y, demo_sensitive, demo_pop, pop_g1, pop_g2, demo_mask_arr)
        #fair_loss = multi_var_fine_grained_diff(prediction, self.y, demo_sensitive, demo_pop, pop_g1, pop_g2, demo_mask_arr)

        # fair_loss = pairwise_fairloss(prediction, self.y, demo_sensitive, demo_pop, demo_mask_arr)
        acc_loss = tf.losses.absolute_difference(prediction, self.y, weight)
        acc_loss = tf.losses.absolute_difference(prediction, self.y, weight)

        # if fairloss_func != None:
        if fairloss_func == "RFG":
            fair_loss = multi_var_mean_diff(prediction, self.y, demo_sensitive, demo_pop, pop_g1, pop_g2, demo_mask_arr)
        if fairloss_func == "IFG":
            fair_loss = multi_var_fine_grained_diff(prediction, self.y, demo_sensitive, demo_pop, pop_g1, pop_g2, demo_mask_arr)
        if fairloss_func == "equalmean":
            fair_loss = equal_mean(prediction, self.y, demo_sensitive, demo_pop, pop_g1, pop_g2,
                        grid_g1, grid_g2, demo_mask_arr)
        if fairloss_func == "pairwise":
            fair_loss = pairwise_fairloss(prediction, self.y, demo_sensitive, demo_pop, demo_mask_arr)
        cost = acc_loss + lamda * fair_loss
        print('fair_loss: ',fair_loss)

        # weighed MAE loss
        '''
        weighted_mae = weighted_MAE_loss(prediction, self.y, demo_sensitive, demo_pop,
                       demo_mask_arr, beta)
        if beta == 1.0:
            cost = acc_loss + lamda * fair_loss
        else:
            cost = weighted_mae + lamda * fair_loss
        '''

        # weighted_mae = differential_weighed_MAE_loss(prediction, self.y, demo_sensitive, demo_pop,
        #               demo_mask_arr, beta)
        # cost = weighted_mae + lamda * fair_loss
        # bin_loss = binary_loss(prediction, self.y, demo_sensitive, demo_pop,
               #        demo_mask_arr)
        # cost = acc_loss + lamda * fair_loss + beta * bin_loss

        # cost = acc_loss
        #cost = acc_loss + lamda * fair_loss
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_input))
        #cost = tf.losses.absolute_difference(prediction, self.y, weight)

        with tf.name_scope("training"):
        # with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = self.global_step)

        saver = tf.train.Saver()
    #     correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        test_result = list()

        if not os.path.exists(save_folder_path):
            os.makedirs(save_path)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # restore latest checkpoint from train_dir

            if checkpoint_path is not None:
                saver.restore(sess, checkpoint_path)
            else:
                saver.restore(sess, tf.train.latest_checkpoint(save_folder_path))
            # check global step
            print("global step: ", sess.run([self.global_step]))
            print("Model restore finished, current globle step: %d" % self.global_step.eval())

            # get new epoch num
            print("int(len(x_train_data) / batch_size +1): ", int(len(x_train_data) / batch_size +1))
            start_epoch_num = tf.div(self.global_step, int(len(x_train_data) / batch_size +1))
            #self.global_step/ (len(x_train_data) / batch_size +1) -1
            print("start_epoch_num: ", start_epoch_num.eval())
            start_epoch = start_epoch_num.eval()

            start_time = datetime.datetime.now()
            if len(x_train_data)%batch_size ==0:
                iterations = int(len(x_train_data)/batch_size)
            else:
                iterations = int(len(x_train_data)/batch_size) + 1
            # run epochs
            # global step = epoch * ( len(x_train_data)/ batchsize +1) + itr

            for epoch in range(start_epoch, epochs):
                start_time_epoch = datetime.datetime.now()
                print('Epoch', epoch, 'started', end='')
                epoch_loss = 0
                epoch_fairloss = 0
                epoch_accloss = 0
                # mini batch
                for itr in range(iterations):
                    mini_batch_x = x_train_data[itr*batch_size: (itr+1)*batch_size]
                    mini_batch_y = y_train_data[itr*batch_size: (itr+1)*batch_size]
                    # model fusion
                    if data_1d_train is not None:
                        mini_batch_data_1d = data_1d_train[itr*batch_size: (itr+1)*batch_size]
                    else:
                        mini_batch_data_1d = None
                    if data_2d_train is not None:
                        mini_batch_data_2d = np.expand_dims(data_2d_train, 0)
                        mini_batch_data_2d = np.tile(mini_batch_data_2d, [mini_batch_x.shape[0], 1,1,1])
                    else:
                        mini_batch_data_2d = None


                    # 1d, 2d, and 3d
                    if data_1d_train is not None and data_2d_train is not None:

                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.input_1d_feature:mini_batch_data_1d,  self.input_2d_feature: mini_batch_data_2d,
                                                            self.is_training: True   })
                    elif data_1d_train is not None:  # 1d and 3d
                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.input_1d_feature:mini_batch_data_1d,
                                                            self.is_training: True   })
                    elif data_2d_train is not None:
                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.input_2d_feature: mini_batch_data_2d,
                                                            self.is_training: True   })
                    else: # only 3d
                        _optimizer, _cost, _acc_loss = sess.run([optimizer, cost, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                                                            self.is_training: True   })




                    # _optimizer, _cost, _fair_loss, _acc_loss = sess.run([optimizer, cost, fair_loss, acc_loss], feed_dict={self.x: mini_batch_x, self.y: mini_batch_y,
                    #                                      self.input_1d_feature:mini_batch_data_1d,  self.input_2d_feature: mini_batch_data_2d,
                    #                                      self.is_training: True   })
                    epoch_loss += _cost
                    # epoch_fairloss += _fair_loss
                    epoch_accloss += _acc_loss

                    if itr % 10 == 0:
                        #print('epoch: {}, step: {}\t\ttrain err: {}'.format(epoch, itr, _cost))
                        print('epoch: {}, step: {}, train err: {}, _fair_loss:{}, mae:{}'.format(epoch, itr, _cost, _fair_loss, _acc_loss))

                # report loss per epoch
                epoch_loss = epoch_loss/ iterations
                epoch_fairloss = epoch_fairloss / iterations
                epoch_accloss = epoch_accloss / iterations

                # train_acc_loss.append(epoch_accloss)
                print('epoch: ', epoch, 'Trainig Set Epoch total Cost: ',epoch_loss)
                print('epoch: ', epoch, 'Trainig Set Epoch fair Cost: ',epoch_fairloss)
                print('epoch: ', epoch, 'Trainig Set Epoch accuracy Cost: ',epoch_accloss)



                #  using mini batch in case not enough memory
                test_cost = 0
                test_acc_loss = 0
                test_fair_loss = 0
                final_output = list()

                print('testing')
                itrs = int(len(x_test_data)/batch_size) + 1
                for itr in range(itrs):
                    mini_batch_x_test = x_test_data[itr*batch_size: (itr+1)*batch_size]
                    mini_batch_y_test = y_test_data[itr*batch_size: (itr+1)*batch_size]
                    # model fusion
                    if data_1d_test is not None:
                        mini_batch_data_1d_test = data_1d_test[itr*batch_size: (itr+1)*batch_size]
                    else:
                        mini_batch_data_1d_test  = None

                    if data_2d_test is not None:
                        mini_batch_data_2d_test = np.expand_dims(data_2d_test, 0)
                        mini_batch_data_2d_test = np.tile(mini_batch_data_2d_test, [mini_batch_x_test.shape[0], 1,1,1])
                    else:
                        mini_batch_data_2d_test = None

                    if data_1d_test is not None and data_2d_test is not None:
                        #acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test ,
                                            self.is_training: True  })
                        test_fair_loss += sess.run(fair_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True})
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True})

                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                                        self.is_training: True})
                    elif data_1d_test is not None:
                        #acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,
                                            self.is_training: True  })
                        test_fair_loss += sess.run(fair_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,
                                            self.is_training: True})
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,
                                            self.is_training: True})

                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_1d_feature:mini_batch_data_1d_test,
                                        self.is_training: True})
                    elif data_2d_test is not None:
                                                #acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True  })
                        test_fair_loss += sess.run(fair_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True})
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True})

                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_2d_feature: mini_batch_data_2d_test,
                                        self.is_training: True})
                    else:
                        test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                          self.is_training: True  })
                        test_fair_loss += sess.run(fair_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.is_training: True})
                        test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.is_training: True})
                        batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.is_training: True})


                    # #acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})
                    # test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                    #                     self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test ,
                    #                     self.is_training: True  })
                    # test_fair_loss += sess.run(fair_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                    #                     self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                    #                     self.is_training: True})
                    # test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                    #                     self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                    #                     self.is_training: True})

                    # batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                    #                  self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                    #                  self.is_training: True})

                    final_output.extend(batch_output)
    #             output = np.array(final_output)

                end_time_epoch = datetime.datetime.now()
                #print(' Testing Set Accuracy:',test_cost/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))
                print(' Testing Set Cost:',test_cost/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))
                print(' Testing Set Fair Cost:',test_fair_loss/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))
                print(' Testing Set Accuracy Cost:',test_acc_loss/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))
                # test_acc_loss.append(test_acc_loss/itrs)


                #save_folder_path  = './fusion_model_'+ str(lamda)+'/'
                # save globel step for resuming training later
                save_path = saver.save(sess, save_folder_path +'fusion_model_' + str(lamda)+'_'+str(epoch)+'.ckpt', global_step=self.global_step)
                print('Model saved to {}'.format(save_path))

                # save epoch statistics to csv
                ecoch_res_df = pd.DataFrame([[epoch_loss, test_cost/itrs, epoch_accloss, test_acc_loss/itrs,
                              epoch_fairloss, test_fair_loss/itrs]],
                    columns=[ 'train_loss','test_loss', 'train_acc', 'test_acc', 'train_fair', 'test_fair'])

                res_csv_path = save_folder_path + 'ecoch_res_df_' + str(lamda)+'.csv'

                with open(res_csv_path, 'a') as f:
                    # Add header if file is being created, otherwise skip it
                    ecoch_res_df.to_csv(f, header=f.tell()==0)

                # save results to txt
                txt_name = save_folder_path + 'fusion_pairwise_df_' +str(lamda)+'.txt'
                with open(txt_name, 'w') as the_file:
                    #the_file.write('Only account for grids that intersect with city boundary \n')
                    the_file.write('epoch\n')
                    the_file.write(str(epoch)+'\n')
                    the_file.write('lamda\n')
                    the_file.write(str(lamda) + '\n')
                    the_file.write(' Testing Set Cost:\n')
                    the_file.write(str(test_cost/itrs) + '\n')
                    the_file.write('Testing Set Fair Cost\n')
                    the_file.write(str(test_fair_loss/itrs)+ '\n')
                    the_file.write('Testing Set Accuracy Cost\n')
                    the_file.write(str(test_acc_loss/itrs)+ '\n')
                    the_file.write('\n')
                    the_file.close()

                if epoch == epochs-1:
                    test_result.extend(final_output)


                # plot results
                print('saving train_test plots')
                train_test = pd.read_csv(save_folder_path  + 'ecoch_res_df_' + str(lamda)+'.csv')
                # train_test = train_test.loc[:, ~train_test.columns.str.contains('^Unnamed')]
                train_test[['train_loss', 'test_loss']].plot()
                plt.savefig(save_folder_path + 'total_loss_inprogress.png')
                train_test[['train_acc', 'test_acc']].plot()
                plt.savefig(save_folder_path + 'acc_loss_inprogress.png')
                train_test[['train_fair', 'test_fair']].plot()
                plt.savefig(save_folder_path + 'fair_loss_inprogress.png')
                plt.close()

            end_time = datetime.datetime.now()
            output = np.array(test_result)
            print('Time elapse: ', str(end_time - start_time))
            return output



    # given test data and checkpoint, do inference
    # TODO: add switch cities and 1d/2d use
    def inference(self, x_test_data, y_test_data, demo_mask_arr,
                    lamda, checkpoint_path,
                    data_2d_test, data_1d_test,demo_sensitive,demo_pop,
                    pop_g1, pop_g2,
                     grid_g1, grid_g2, fairloss_func,
                      save_folder_path, beta,
                      keep_rate=0.7,
                    batch_size=64):
        # tf.reset_default_graph()
        test_result = list()
        final_output = list()

            # demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
            # with tf.name_scope('inputs'):
            # # [batchsize, depth, height, width, channel]
            #     x_input = tf.placeholder(tf.float32, shape=[None,time_steps, height, width, channel])
            #     #
            #     #y_input = tf.placeholder(tf.float32, shape=[None, n_classes])
            #     y_input = tf.placeholder(tf.float32, shape= [None, height, width, channel])

        #     with tf.name_scope("rmse"):
            #prediction = cnn_model(x_input)
            #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_input))

            #weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)

            # customize loss function
            #fair_loss = group_fairloss(prediction, y_input, demo_sensitive, demo_pop)

            #fair_loss = mean_diff(prediction, y_input, demo_sensitive, demo_pop, pop_g1, pop_g2, demo_mask_arr)

            #fair_loss = multi_var_mean_diff(prediction, y_input, demo_sensitive, demo_pop, pop_g1, pop_g2, demo_mask_arr)
        #     fair_loss = multi_var_fine_grained_diff(prediction, y_input, demo_sensitive, demo_pop, pop_g1, pop_g2, demo_mask_arr)
            #acc_loss = tf.losses.absolute_difference(prediction, y_input, weight)
        #     print('fair_loss: ',fair_loss)
            #cost = acc_loss
        #     cost = acc_loss + lamda * fair_loss
            #cost = tf.losses.absolute_difference(prediction, y_input, weight)

                   #prediction = self.cnn_model(self.x, keep_rate, seed=1)
            # fusion model
        prediction_3d = self.cnn_model(self.x, self.is_training, keep_rate, seed=1)

        if data_2d_test is None:
            prediction_2d = None
                # 32,20,1
                #prediction_2d = self.cnn_2d_msodel(self.input_2d_feature)
        else:
            prediction_2d = self.cnn_2d_model(self.input_2d_feature, self.is_training, )

        if data_1d_test is None:
                # batchsize, 1,1,1
            prediction_1d = None
                #prediction_1d = self.cnn_1d_model(self.input_1d_feature)
        else:
            prediction_1d = self.cnn_1d_model(self.input_1d_feature, self.is_training, )
                #prediction_1d = None

            # fusion
        prediction = self.model_fusion(prediction_3d, prediction_2d, prediction_1d, self.is_training, )

        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
            # [1, 32, 20, 1] -> [batchsize, 32, 20, 1]
            # batchsize = tf.shape(prediction)[0]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(prediction)[0],1,1,1])
        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)
            #fair_loss = group_fairloss(prediction, y_input, demo_sensitive, demo_pop)
            #fair_loss = mean_diff(prediction, self.y, demo_sensitive, demo_pop, pop_g1, pop_g2, demo_mask_arr)
            # multi_var fairness loss: input multi_demo_sensitive shape : [32, 20, 3]
            #fair_loss = multi_var_mean_diff(prediction, self.y, demo_sensitive, demo_pop, pop_g1, pop_g2, demo_mask_arr)
            #fair_loss = multi_var_fine_grained_diff(prediction, self.y, demo_sensitive, demo_pop, pop_g1, pop_g2, demo_mask_arr)

        # fair_loss = pairwise_fairloss(prediction, self.y, demo_sensitive, demo_pop, demo_mask_arr)
        acc_loss = tf.losses.absolute_difference(prediction, self.y, weight)
        # if fairloss_func != None:
        if fairloss_func == "RFG":
            fair_loss = multi_var_mean_diff(prediction, self.y, demo_sensitive, demo_pop, pop_g1, pop_g2, demo_mask_arr)
        if fairloss_func == "IFG":
            fair_loss = multi_var_fine_grained_diff(prediction, self.y, demo_sensitive, demo_pop, pop_g1, pop_g2, demo_mask_arr)
        if fairloss_func == "equalmean":
            fair_loss = equal_mean(prediction, self.y, demo_sensitive, demo_pop, pop_g1, pop_g2,
                        grid_g1, grid_g2, demo_mask_arr)
        if fairloss_func == "pairwise":
            fair_loss = pairwise_fairloss(prediction, self.y, demo_sensitive, demo_pop, demo_mask_arr)
        cost = acc_loss + lamda * fair_loss
        print('fair_loss: ',fair_loss)

        # weighed MAE loss
        '''
        weighted_mae = weighted_MAE_loss(prediction, self.y, demo_sensitive, demo_pop,
                       demo_mask_arr, beta)
        if beta == 1.0:
            cost = acc_loss + lamda * fair_loss
        else:
            cost = weighted_mae + lamda * fair_loss
        '''

        # weighted_mae = differential_weighed_MAE_loss(prediction, self.y, demo_sensitive, demo_pop,
        #               demo_mask_arr, beta)

        # bin_loss = binary_loss(prediction, self.y, demo_sensitive, demo_pop,
                    #    demo_mask_arr)
        # cost = acc_loss + lamda * fair_loss + beta * bin_loss


       #cost = weighted_mae + lamda * fair_loss

        cost = acc_loss
        # cost = acc_loss + lamda * fair_loss

            #saver = tf.train.Saver(restore_vars, name='ema_restore')
        #     saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
        saver = tf.train.Saver()
        #     new_saver = tf.train.import_meta_graph("3d_cnn_model_metric1_0.75_119.ckpt.meta")

        test_cost = 0
        test_acc_loss = 0
        test_fair_loss = 0

        # global_step = tf.train.get_or_create_global_step()

        with tf.Session() as sess:
        #         tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint_path)
            # check global step
            print("global step: ", sess.run([self.global_step]))
            print("Model restore finished, current globle step: %d" % self.global_step.eval())

            print('testing')
            itrs = int(len(x_test_data)/batch_size) + 1
            for itr in range(itrs):
                mini_batch_x_test = x_test_data[itr*batch_size: (itr+1)*batch_size]
                mini_batch_y_test = y_test_data[itr*batch_size: (itr+1)*batch_size]
                    # model fusion
                if data_1d_test is not None:
                    mini_batch_data_1d_test = data_1d_test[itr*batch_size: (itr+1)*batch_size]
                else:
                    mini_batch_data_1d_test  = None

                if data_2d_test is not None:
                    mini_batch_data_2d_test = np.expand_dims(data_2d_test, 0)
                    mini_batch_data_2d_test = np.tile(mini_batch_data_2d_test, [mini_batch_x_test.shape[0], 1,1,1])
                else:
                    mini_batch_data_2d_test = None


                if data_1d_test is not None and data_2d_test is not None:
                        #acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})
                    test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test ,
                                            self.is_training: True  })
                    test_fair_loss += sess.run(fair_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True})
                    test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True})

                    batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                                        self.is_training: True})
                elif data_1d_test is not None:
                        #acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})
                    test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,
                                            self.is_training: True  })
                    test_fair_loss += sess.run(fair_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,
                                            self.is_training: True})
                    test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_1d_feature:mini_batch_data_1d_test,
                                            self.is_training: True})

                    batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_1d_feature:mini_batch_data_1d_test,
                                        self.is_training: True})
                elif data_2d_test is not None:
                                                #acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})
                    test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True  })
                    test_fair_loss += sess.run(fair_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True})
                    test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.input_2d_feature: mini_batch_data_2d_test,
                                            self.is_training: True})

                    batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.input_2d_feature: mini_batch_data_2d_test,
                                        self.is_training: True})
                else:
                    test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                          self.is_training: True  })
                    test_fair_loss += sess.run(fair_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.is_training: True})
                    test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                            self.is_training: True})
                    batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                                        self.is_training: True})


                    #acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})
                # test_cost += sess.run(cost, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                #                         self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test ,
                #                         self.is_training: True  })
                # test_fair_loss += sess.run(fair_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                #                         self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                #                         self.is_training: True})
                # test_acc_loss += sess.run(acc_loss, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                #                         self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                #                         self.is_training: True})

                # batch_output = sess.run(prediction, feed_dict={self.x: mini_batch_x_test, self.y: mini_batch_y_test,
                #                      self.input_1d_feature:mini_batch_data_1d_test,  self.input_2d_feature: mini_batch_data_2d_test,
                #                      self.is_training: True})
                final_output.extend(batch_output)

            print(' Testing Set Cost:',test_cost/itrs)
            print(' Testing Set Fair Cost:',test_fair_loss/itrs)
            print(' Testing Set Accuracy Cost:',test_acc_loss/itrs)
            output = np.array(final_output)

        return output






'''
fixed lenght time window: 168 hours
no exogenous features
'''
class Conv3D:
    def __init__(self, train_obj, train_arr, test_arr, intersect_pos_set,
                    # demo_sensitive, demo_pop, pop_g1, pop_g2,
                    # grid_g1, grid_g2, fairloss,
                    train_arr_1d, test_arr_1d, data_2d,
                     demo_mask_arr,
                     save_path,
                     HEIGHT, WIDTH, TIMESTEPS, BIKE_CHANNEL,
                     NUM_2D_FEA, NUM_1D_FEA, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                     is_inference = False, checkpoint_path = None,
                     resume_training = False, train_dir = None
                     ):
                     #  if s_inference = True, do inference only
        self.train_obj = train_obj
        self.train_df = train_obj.train_df
        self.test_df = train_obj.test_df
        self.train_arr = train_arr
        self.test_arr = test_arr

        self.intersect_pos_set = intersect_pos_set
        self.demo_mask_arr = demo_mask_arr


        self.train_arr_1d = train_arr_1d
        self.test_arr_1d = test_arr_1d
        self.data_2d = data_2d
        self.save_path = save_path

        globals()['HEIGHT']  = HEIGHT
        globals()['WIDTH']  = WIDTH
        globals()['TIMESTEPS']  = TIMESTEPS
        globals()['BIKE_CHANNEL']  = BIKE_CHANNEL
        globals()['NUM_2D_FEA']  = NUM_2D_FEA
        globals()['NUM_1D_FEA']  = NUM_1D_FEA
        globals()['BATCH_SIZE']  = BATCH_SIZE
        globals()['TRAINING_STEPS']  = TRAINING_STEPS
        globals()['LEARNING_RATE']  = LEARNING_RATE

        print('Conv3D recieved: ')
        print('HEIGHT: ', HEIGHT)
        print('start learning rate: ',LEARNING_RATE)

        # print("self.data_2d: shape: ", self.data_2d.shape)

        self.is_inference = is_inference
        self.checkpoint_path = checkpoint_path

        self.resume_training = resume_training
        self.train_dir = train_dir

        # ignore non-intersection cells in test_df
        # this is for evaluation
        self.test_df_cut = self.test_df.loc[:,self.test_df.columns.isin(list(self.intersect_pos_set))]

        if is_inference == False:
            if resume_training == False:
                # get prediction results
                print('training from scratch, and get prediction results')
                # predicted_vals: (552, 30, 30, 1)
                self.predicted_vals = self.run_conv3d()
                np.save(self.save_path +'prediction_arr.npy', self.predicted_vals)
            else:
                # resume training
                print("resume training, and get prediction results")
                self.predicted_vals  = self.run_resume_training()
                np.save(self.save_path +'resumed_prediction_arr.npy', self.predicted_vals)

        else:
            # inference only
            print('get inference results')
            self.predicted_vals  = self.run_inference()
            np.save(self.save_path +'inference_arr.npy', self.predicted_vals)


        # calculate performance using only cells that intersect with city boundary
        # do evaluation using matrix format
        self.evaluation()

        # convert predicted_vals to pandas dataframe with timestamps
        self.conv3d_predicted = self.arr_to_df()


    # run training and testing together
    def run_conv3d(self):
        tf.reset_default_graph()
        # self, channel, time_steps, height, width
        predictor = Conv3DPredictor(self.intersect_pos_set,
                                # self.demo_sensitive, self.demo_pop,
                                #     self.pop_g1, self.pop_g2,self.grid_g1, self.grid_g2, self.fairloss,
                                    # self.data_2d, self.data_1d.X, self.data_2d, self.data_1d_test.X,
                                     self.demo_mask_arr, channel=BIKE_CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH,
                                    )
        #data = data_loader.load_series('international-airline-passengers.csv')
        # rawdata, timesteps, batchsize
        self.train_data = generateData_3d_feature(self.train_arr, TIMESTEPS, BATCH_SIZE)
        # train_x shape should be : [num_examples (batch size), time_step (168), feature_dim (1)]

        # create batches, feed batches into predictor
        # predictor.train(self.train_data)
        # print('finished training')

        # prep test data, split test_arr into test_x and test_y
        self.test_data = generateData_3d_feature(self.test_arr, TIMESTEPS, BATCH_SIZE)
        print('test_data.y.shape', self.test_data.y.shape)

        if self.train_arr_1d is not None:
            self.train_data_1d = generateData_1d(self.train_arr_1d, TIMESTEPS, BATCH_SIZE)
            self.test_data_1d = generateData_1d(self.test_arr_1d, TIMESTEPS, BATCH_SIZE)
            print('test_data_1d.y.shape', self.test_data_1d.y.shape)
            predicted_vals = predictor.train_neural_network(self.train_data.X, self.train_data.y,
                        self.test_data.X, self.test_data.y,
                        # self.demo_sensitive, self.demo_pop, self.pop_g1, self.pop_g2,
                        #  self.grid_g1, self.grid_g2, self.fairloss,
                         self.demo_mask_arr,
                        self.data_2d, self.train_data_1d.X, self.data_2d, self.test_data_1d.X,
                          self.save_path,
                          # self.beta,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        else:
            print('No 1d feature')
            self.train_data_1d = None
            self.test_data_1d = None
            predicted_vals = predictor.train_neural_network(self.train_data.X, self.train_data.y,
                        self.test_data.X, self.test_data.y,
                        # self.demo_sensitive, self.demo_pop, self.pop_g1, self.pop_g2,
                        # self.grid_g1, self.grid_g2, self.fairloss,
                     self.demo_mask_arr,
                        self.data_2d, None, self.data_2d, None,
                          self.save_path,
                          # self.beta,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)



        # with tf.Session() as sess:
        #     predicted_vals = predictor.test(sess, self.test_data)
        #    print('predicted_vals', np.shape(predicted_vals))


        # predicted_vals = predictor.train_neural_network(self.train_data.X, self.train_data.y,
        #                 self.test_data.X, self.test_data.y,
        #                 self.demo_sensitive, self.demo_pop, self.pop_g1, self.pop_g2, self.lamda, self.demo_mask_arr,
        #                 self.data_2d, self.train_data_1d.X, self.data_2d, self.test_data_1d.X,
        #                   self.save_path,
        #                   self.beta,
        #              epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)


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


    # run training and testing together
    def run_resume_training(self):
        tf.reset_default_graph()
        # self, channel, time_steps, height, width
        predictor = Conv3DPredictor(self.intersect_pos_set, self.demo_sensitive, self.demo_pop,
                                    self.pop_g1, self.pop_g2,self.grid_g1, self.grid_g2, self.fairloss,
                                    # self.data_2d, self.data_1d.X, self.data_2d, self.data_1d_test.X,
                                     self.lamda, self.demo_mask_arr, channel=BIKE_CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH,
                                    )
        #data = data_loader.load_series('international-airline-passengers.csv')
        # rawdata, timesteps, batchsize
        self.train_data = generateData_3d_feature(self.train_arr, TIMESTEPS, BATCH_SIZE)
        # train_x shape should be : [num_examples (batch size), time_step (168), feature_dim (1)]
        # prep test data, split test_arr into test_x and test_y
        self.test_data = generateData_3d_feature(self.test_arr, TIMESTEPS, BATCH_SIZE)
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
    def run_inference(self):
        tf.reset_default_graph()
        # self, channel, time_steps, height, width
        predictor = Conv3DPredictor(self.intersect_pos_set, self.demo_sensitive, self.demo_pop,
                                    self.pop_g1, self.pop_g2,self.grid_g1, self.grid_g2, self.fairloss,
                                    # self.data_2d, self.data_1d.X, self.data_2d, self.data_1d_test.X,
                                     self.lamda, self.demo_mask_arr, channel=BIKE_CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH,
                                    )
        #data = data_loader.load_series('international-airline-passengers.csv')
        # rawdata, timesteps, batchsize
       #self.train_data = generateData(self.train_arr, TIMESTEPS, BATCH_SIZE)
        # train_x shape should be : [num_examples (batch size), time_step (168), feature_dim (1)]

        # create batches, feed batches into predictor
        # predictor.train(self.train_data)
        # print('finished training')

        # prep test data, split test_arr into test_x and test_y
        self.test_data = generateData_3d_feature(self.test_arr, TIMESTEPS, BATCH_SIZE)
        print('test_data.y.shape', self.test_data.y.shape)

        if self.test_arr_1d is not None:
            # self.train_data_1d = generateData_1d(self.train_arr_1d, TIMESTEPS, BATCH_SIZE)
            self.test_data_1d = generateData_1d(self.test_arr_1d, TIMESTEPS, BATCH_SIZE)
            print('test_data_1d.y.shape', self.test_data_1d.y.shape)
            predicted_vals = predictor.inference(self.test_data.X, self.test_data.y,  self.demo_mask_arr,
                    self.lamda, self.checkpoint_path,
                    self.data_2d, self.test_data_1d.X, self.demo_sensitive,self.demo_pop,
                    self.pop_g1, self.pop_g2,
                        self.grid_g1, self.grid_g2, self.fairloss,
                      self.save_path, self.beta,
                    batch_size=BATCH_SIZE)
        else:
            print('No 1d feature')
            # self.train_data_1d = None
            self.test_data_1d = None
            predicted_vals = predictor.inference(self.test_data.X, self.test_data.y,  self.demo_mask_arr,
                    self.lamda, self.checkpoint_path,
                    self.data_2d, None, self.demo_sensitive,self.demo_pop,
                    self.pop_g1, self.pop_g2,
                        self.grid_g1, self.grid_g2, self.fairloss,
                      self.save_path, self.beta,
                    batch_size=BATCH_SIZE)


        # with tf.Session() as sess:
        #     predicted_vals = predictor.test(sess, self.test_data)
        #    print('predicted_vals', np.shape(predicted_vals))


        # predicted_vals = predictor.train_neural_network(self.train_data.X, self.train_data.y,
        #                 self.test_data.X, self.test_data.y,
        #                 self.demo_sensitive, self.demo_pop, self.pop_g1, self.pop_g2, self.lamda, self.demo_mask_arr,
        #                 self.data_2d, self.train_data_1d.X, self.data_2d, self.test_data_1d.X,
        #                   self.save_path,
        #              epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)




        predicted = predicted_vals.flatten()
        y = self.test_data.y.flatten()
        #mse = mean_absolute_error(y['test'], predicted)
        #print ("Error: %f" % mse)
        rmse = np.sqrt((np.asarray((np.subtract(predicted, y))) ** 2).mean())
        mae = mean_absolute_error(predicted, y)
        print('INFERENCE results')
        print('Metrics for all grids: ')
        print("RSME: %f" % rmse)
        print('MAE: %f' %mae)
        return predicted_vals




    # evaluate rmse and mae with grids that intersect with city boundary
    def evaluation(self):
        sample_pred_squeeze = np.squeeze(self.predicted_vals)
        test_squeeze = np.squeeze(self.test_data.y)
        pred_shape = self.predicted_vals.shape
        mse = 0
        mae = 0
        mape = 0
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
                        if test_rot[r][c]!=0:
                            mape += abs(test_rot[r][c] - temp_rot[r][c]) / test_rot[r][c]

        rmse = math.sqrt(mse / (pred_shape[0] * len(self.intersect_pos_set)))
        mae = mae / (pred_shape[0] * len(self.intersect_pos_set))
        mape = mape/ (pred_shape[0] * len(self.intersect_pos_set))
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
        print('mape: ', mape)
        print('count ', count)

    # convert predicted result tensor back to pandas dataframe
    def arr_to_df(self):
        convlstm_predicted = pd.DataFrame(np.nan,
                                index=self.test_df_cut[self.train_obj.predict_start_time: self.train_obj.predict_end_time].index,
                                columns=list(self.test_df_cut))

        sample_pred_squeeze = np.squeeze(self.predicted_vals)
        pred_shape = self.predicted_vals.shape

        # loop through time stamps
        for i in range(0, pred_shape[0]):
            temp_image = sample_pred_squeeze[i]
            # test_image = test_squeeze[i]
            # rotate
            temp_rot = np.rot90(temp_image, axes=(1,0))
        #     test_rot= np.rot90(test_image, axes=(1,0))

            dt = datetime_utils.str_to_datetime(self.train_obj.test_start_time) + datetime.timedelta(hours=i * 3)
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
