# TOY CASE

# v3: datasets were grouped during encoding and decoding
# according to a predefined grouping strategy
# [raw datasets grouping by Pearson correlation and affinity propogation ]
#
# train autoencoder for urban features
# for each week's data, learn a
# laten representation as [H, W, dim]
# from the latent representation, each datasets will
# be reconstructed with equal weight in MAE loss


# last updated: Jan 7 2020
# add variable_scopes to variables to restore
# change saved model name and path
# added restoring some variables from pretrained all-to-all AE

# last updated: Jan 8 2020
# fix pretrained weights
# output second level encoded layers

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

import tensorflow.python.keras
import tensorflow.contrib.keras as keras
from tensorflow.python.keras import backend as K

import pickle


HEIGHT = 32
WIDTH = 20
TIMESTEPS = 168

BATCH_SIZE = 32
# actually epochs
TRAINING_STEPS = 50
# TRAINING_STEPS = 1
LEARNING_RATE = 0.003
HOURLY_TIMESTEPS = 168
DAILY_TIMESTEPS = 7
THREE_HOUR_TIMESTEP = 56


def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)


def generate_fixlen_timeseries(rawdata_arr, timestep = 168):
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


# create sequences in real time
def create_mini_batch_1d(start_idx, end_idx,  data_1d):
    # data_3d : (45984, 32, 20, ?)
    # data_1d: (45984, ?)
    # data_2d: (32, 20, ?)
    test_size = end_idx - start_idx

    test_data_1d = data_1d[start_idx:end_idx + 168 - 1,:]
    test_data_1d_seq = generate_fixlen_timeseries(test_data_1d)
    test_data_1d_seq = np.swapaxes(test_data_1d_seq,0,1)
    # (168, batchsize, dim)
    return test_data_1d_seq

# output: batchsize, h, w, dim
def create_mini_batch_2d(start_idx, end_idx,  data_2d):
    # data_3d : (45984, 32, 20, ?)
    # data_1d: (45984, ?)
    # data_2d: (32, 20, ?)
    test_size = end_idx - start_idx
    test_data_2d = np.expand_dims(data_2d, axis=0)
    test_data_2d = np.tile(test_data_2d,(test_size,1,1,1))
    # (batchsize, 32, 20, 20)
    return test_data_2d

# note: 3d data has different time steps
# if hourly, output [batchsize, 168, w, h, dim]
# iif 24-hour, output [batchsize, 7, w, h, dim]
'''
(1916, 32, 20)
(1916, 32, 20)
(45984, 32, 20)
'''
def create_mini_batch_3d(start_idx, end_idx,data_3d, timestep):
    # data_3d : (45984, 32, 20, ?)
    # data_1d: (45984, ?)
    # data_2d: (32, 20, ?)

    test_size = end_idx - start_idx
    # handle different time frame
    # shape should be (batchsize, 7, 32, 20, 1), but for 24 hours in a day
    # the sequence should be the same.
    if timestep == 7:
        # (7, 45840, 32, 20)
        test_data_3d_seq = data_3d[:, start_idx :end_idx, :, :]
        test_data_3d_seq = np.expand_dims(test_data_3d_seq, axis=4)
        test_data_3d_seq = np.swapaxes(test_data_3d_seq,0,1)
    else:
        test_data_3d = data_3d[start_idx :end_idx + timestep - 1, :, :]
        test_data_3d_seq = generate_fixlen_timeseries(test_data_3d, timestep)
        test_data_3d_seq = np.expand_dims(test_data_3d_seq, axis=4)
        test_data_3d_seq = np.swapaxes(test_data_3d_seq,0,1)
    # (timestep (168/56/7), batchsize, 32, 20, 1)
    return test_data_3d_seq


# get variables to be restored from pretrained model
def get_variables_to_restore(variables, scopes_to_reserve):
    variables_to_restore = []
    for v in variables:
        if v.name.split(':')[0].split('/')[0] in scopes_to_reserve:
            print("Variables restored: %s" % v.name)
            variables_to_restore.append(v)

    return variables_to_restore



# get names of variable_scopes to be restored
def get_scopes_to_restore(rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict):
    keys_1d = list(rawdata_1d_dict.keys())
    keys_2d = list(rawdata_2d_dict.keys())
    keys_3d = list(rawdata_3d_dict.keys())
    scopes_to_reserve = []
    prefix_list = ['1d_data_process_', '2d_data_process_', '3d_data_process_']
    suffix_list = keys_1d + keys_2d + keys_3d
    for prefix in prefix_list:
        for suffix in suffix_list:
            scopes_to_reserve.append(prefix + suffix)
    return scopes_to_reserve


class Autoencoder:
    # input_dim = 1, seq_size = 168,
    def __init__(self, rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
                   intersect_pos_set,
                    demo_mask_arr, dim,grouping_dict,
                    channel, time_steps, height, width):

        self.time_steps = time_steps
        self.width = width
        self.height = height
        self.channel = channel  # 27
        self.dim  = dim # default = 1, it is the dimension of latent representation
        self.grouping_dict = grouping_dict
        # this is usefor Batch normalization.
        # https://towardsdatascience.com/pitfalls-of-batch-norm-in-tensorflow-and-sanity-checks-for-training-networks-e86c207548c8
        self.is_training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, trainable=False)
        self.dataset_keys = list(rawdata_1d_dict.keys()) + list(rawdata_2d_dict.keys()) + list(rawdata_3d_dict.keys())

        self.rawdata_1d_tf_x_dict = {}
        self.rawdata_1d_tf_y_dict = {}
        # rawdata_1d_dict
        for k, v in rawdata_1d_dict.items():
            dim = v.shape[-1]
            self.rawdata_1d_tf_x_dict[k] = tf.placeholder(tf.float32, shape=[None,168, dim])
            self.rawdata_1d_tf_y_dict[k] = tf.placeholder(tf.float32, shape=[None,168, dim])

        # 2d
        self.rawdata_2d_tf_x_dict = {}
        self.rawdata_2d_tf_y_dict = {}
        # rawdata_1d_dict
        for k, v in rawdata_2d_dict.items():
            dim = v.shape[-1]
            self.rawdata_2d_tf_x_dict[k] = tf.placeholder(tf.float32, shape=[None, height, width, dim])
            self.rawdata_2d_tf_y_dict[k] = tf.placeholder(tf.float32, shape=[None, height, width, dim])

        # -------- 3d --------------#
        # building_permit_x = tf.placeholder(tf.float32, shape=[None,DAILY_TIMESTEPS, height, width, 1])
        # building_permit_y = tf.placeholder(tf.float32, shape=[None,DAILY_TIMESTEPS, height, width, 1])
        # collisions_x = tf.placeholder(tf.float32, shape=[None,DAILY_TIMESTEPS, height, width, 1])
        # collisions_y = tf.placeholder(tf.float32, shape=[None,DAILY_TIMESTEPS, height, width, 1])
        # seattle911calls_x = tf.placeholder(tf.float32, shape=[None,HOURLY_TIMESTEPS, height, width, 1])
        # seattle911calls_y = tf.placeholder(tf.float32, shape=[None,HOURLY_TIMESTEPS, height, width, 1])
        #
        # self.rawdata_3d_tf_x_dict = {
        #     'building_permit': building_permit_x,
        #     'collisions': collisions_x,
        #     'seattle911calls': seattle911calls_x
        #
        # }
        # self.rawdata_3d_tf_y_dict = {
        #       'building_permit': building_permit_y,
        #     'collisions': collisions_y,
        #     'seattle911calls': seattle911calls_y
        # }





    def cnn_model(self, x_train_data, is_training, suffix = '',  output_dim = 3, keep_rate=0.7, seed=None):
        # output from 3d cnn (?, 168, 32, 20, 1)  * weight + b = (?, 32, 20, 1)
        var_scope = "3d_data_process_" + suffix
        with tf.variable_scope(var_scope):
            # conv => 16*16*16
            # input: (?, 168, 32, 20, 2)
            conv1 = tf.layers.conv3d(inputs=x_train_data, filters=16, kernel_size=[3,3,3], padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)
            # conv => 16*16*16
            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)
            # pool => 8*8*8
            #pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
            conv3 = tf.layers.conv3d(inputs=conv2, filters=1, kernel_size=[3,3,3], padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

        # transfer (?, 168, 32, 20, 1) to (?,  32, 20, 168)
        # squeeze -> (?, 168, 32, 20)
            cnn3d_bn_squeeze = tf.squeeze(conv3, axis = 4)
            # swap axes -> (?, 32, 20, 168) -> [0, 1, 2, 3] -> []
            cnn3d_bn_squeeze = tf.transpose(cnn3d_bn_squeeze, perm=[0,2,3, 1])

        # output should be (?, 32, 20, 1)
            conv5 = tf.layers.conv2d(
                      inputs=cnn3d_bn_squeeze,
                      filters=output_dim,
                      kernel_size=[1, 1],
                      padding="same",
                      activation=my_leaky_relu
                      #reuse = tf.AUTO_REUSE
                )
            #
            out = conv5
        # output size should be [None, height, width, channel]
        return out



    '''
    input: 2d feature tensor: height * width * # of features (batchsize, 32, 20, 4)
    output: (32, 20, 1)
    '''
    def cnn_2d_model(self, x_2d_train_data, is_training, suffix = '', output_dim = 1, seed=None):
        var_scope = "2d_data_process_" + suffix
        with tf.variable_scope(var_scope):
            '''
            # conv => 16*16*16
            conv1 = tf.layers.conv3d(inputs=x_train_data, filters=16, kernel_size=[3,3,3], padding='same', activation=my_leaky_relu)
            # conv => 16*16*16
            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=my_leaky_relu)
            # pool => 8*8*8
            #pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
            conv3 = tf.layers.conv3d(inputs=conv2, filters=1, kernel_size=[3,3,3], padding='same', activation=my_leaky_relu)
            '''
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(x_2d_train_data, 16, 3, padding='same',activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            #  Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 32, 3, padding='same',activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

        # output should be (?, 32, 20, 1)
        # with tf.name_scope("2d_layer_b"):
            conv3 = tf.layers.conv2d(
                      inputs=conv2,
                      filters=output_dim,
                      kernel_size=[1, 1],
                      padding="same",
                      activation=my_leaky_relu
                      #reuse = tf.AUTO_REUSE
                )
            out = conv3
        # output size should be [height, width, 1]
        return out


    '''
    input: 1d feature tensor: height * width * # of features
                (batchsize, # of timestamp, channel), e.g., (32, 168,  3)
    output: (batchsize, 1)
    '''
    # (batchsize, 168, # of features)
    def cnn_1d_model(self, x_1d_train_data, is_training, suffix = '', output_dim =3, seed=None):
        var_scope = "1d_data_process_" + suffix
        with tf.variable_scope(var_scope):
            # https://www.tensorflow.org/api_docs/python/tf/layers/conv1d
            '''
            # conv => 16*16*16
            conv1 = tf.layers.conv3d(inputs=x_train_data, filters=16, kernel_size=[3,3,3], padding='same', activation=my_leaky_relu)
            # conv => 16*16*16
            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=my_leaky_relu)
            # pool => 8*8*8
            #pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
            conv3 = tf.layers.conv3d(inputs=conv2, filters=1, kernel_size=[3,3,3], padding='same', activation=my_leaky_relu)
            '''
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv1d(x_1d_train_data, 16, 3, padding='same',activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            #  Convolution Layer with 64 filters and a kernel size of 3
            # output shape: None, 168,16
            conv2 = tf.layers.conv1d(conv1, 32, 3,padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            # Average Pooling   None, 168,16  -> None, 1, 16
            conv2 = tf.layers.average_pooling1d( conv2, 168, 1, padding='valid')

        # with tf.name_scope("1d_layer_b"):
            conv3 = tf.layers.conv1d(
                      inputs=conv2,
                      filters=output_dim,
                      kernel_size=1,
                      padding="same",
                      activation=my_leaky_relu
                      #reuse = tf.AUTO_REUSE
                )

            # squeeze  None, 1, 1  -> None, 1
            conv3_squeeze = tf.squeeze(conv3, axis = 1)
            out = conv3_squeeze
            print('model 1d cnn output :',out.shape )
            # output size should be [None, 1],
        return out



    def model_fusion(self, med_res_3d, med_res_2d, med_res_1d, dim, is_training):
        # prediction_1d: batchsize, 1  -> duplicate to batch size, 32, 20, 1
        temp_list = []

        print('check med_res_1d: ')
        for prediction_1d in med_res_1d:
    #         print('prediction_1d.shape: ', prediction_1d.shape)
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            prediction_1d_expand = tf.tile(prediction_1d, [1, HEIGHT,
                                                    WIDTH ,1])

            temp_list.append(prediction_1d_expand)

        print('check med_res_2d: ')
        for prediction_2d in med_res_2d:
            temp_list.append(prediction_2d)

        print('check med_res_3d:')
        for prediction_3d in med_res_3d:
    #         print('prediction_3d.shape: ', prediction_3d.shape)
            temp_list.append(prediction_3d)

        fuse_feature =tf.concat(axis=3,values=temp_list)
        print('fuse_feature.shape: ', fuse_feature.shape)

        with tf.name_scope("fusion_layer_a"):
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(fuse_feature, 16, 3, padding='same',activation=my_leaky_relu)
        with tf.name_scope("fusion_batch_norm"):
            cnn2d_bn = tf.layers.batch_normalization(inputs=conv1, training=is_training)
            # (?, 168, 32, 20, 1)
            print('cnn2d_bn shape: ',cnn2d_bn.shape)


        # output should be (?, 32, 20, 1)
        with tf.name_scope("fusion_layer_b"):
            conv3 = tf.layers.conv2d(
                      inputs=cnn2d_bn,
                      filters=dim,
                      kernel_size=[1, 1],
                      padding="same",
                      activation=my_leaky_relu
                      #reuse = tf.AUTO_REUSE
                )
        #
        out = conv3
        print('latent representation shape: ',out.shape)
        # output size should be [batchsize, height, width, dim]
        return out


    # [batchsize, height, width, dim] -> recontruct to [None, DAILY_TIMESTEPS, height, width, 1]
    def reconstruct_3d(self, latent_fea, timestep):
        padding = 'SAME'
        stride = [1,1,1]
        # [batchsize, 32, 20, dim] -> [batchsize, 1, 32, 20, dim]
        latent_fea = tf.expand_dims(latent_fea, 1)
        if timestep == 168:
            # [batchsize, 32, 20, dim]-> [batchsize, 168, 32, 20, 1]
            deconv1 = tf.layers.conv3d_transpose(inputs=latent_fea, filters=16, kernel_size=(3,3,3), padding= padding , strides = stride, activation=my_leaky_relu)
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
                    # [168, 32, 20, 9]
            output = tf.layers.conv3d(inputs=unpool4, filters= 1, kernel_size=[3,3,3], padding='same', activation=my_leaky_relu)
                    # [none, 1, 20, 64, 64] -> [none, 20, 64, 64, 1]
            # (?, 168, 32, 20, 9)
            print('output reconstruction 3d shape: ', output.shape)
            return output

        if timestep == 7:
            # [1, 32, 20, 1]-> [168, 32, 20, 9]
            deconv1 = tf.layers.conv3d_transpose(inputs=latent_fea, filters=32, kernel_size=(3,3,3), padding= padding , strides = stride, activation=my_leaky_relu)
            # [1, 32, 20, 32]
            # https://www.tensorflow.org/api_docs/python/tf/keras/backend/resize_volumes
            unpool1 = K.resize_volumes(deconv1,7,1,1,"channels_last")
            # [7, 32, 20, 32]
                    # [168, 32, 20, 9]
            output = tf.layers.conv3d(inputs=unpool1, filters= 1, kernel_size=[3,3,3], padding='same', activation=my_leaky_relu)
                    # [none, 1, 20, 64, 64] -> [none, 20, 64, 64, 1]
            # (?, 168, 32, 20, 9)
            print('output reconstruction 3d shape: ', output.shape)
            return output

    # [None, 32, 20, dim ] -> recontruct to data_2d: (None, 32, 20, dim_2d)
    def reconstruct_2d(self, latent_fea, dim_2d, is_training):
        padding = 'SAME'
        stride = [1, 1]
        #  [None, 32, 20, dim] - > [None, 32, 20, 16]
        conv1 = tf.layers.conv2d(latent_fea, 16, 3, padding='same',activation=None)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

        conv2 = tf.layers.conv2d(conv1, 32, 3, padding='same',activation=None)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)
        # [None, 32, 20, 16]  -> [None,32, 20 32]

        conv3 = tf.layers.conv2d(conv2, dim_2d, 3, padding='same',activation=None)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)
        #[None,32, 20 32] -> [None, 32, 20, dim_2d]
        return conv3


        # [None, 32, 20, dim ] -> recontruct to [None,168, dim_1d]
    def reconstruct_1d(self, latent_fea, dim_1d, is_training):
        padding = 'SAME'
        stride = [1]
        # first: [None, 32, 20, dim] -> [1, 1, dim_1d]
        # then [None, 32, 20, dim] - > [None, 1, 1, 168]
        conv1 = tf.layers.conv2d(latent_fea, 16, 3, padding='same',activation=None)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)
        # [None, 32, 20, dim]  -> [None, 16, 10, 16]
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)


        conv2 = tf.layers.conv2d(conv1, 32, 3, padding='same',activation=None)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)
        # [None, 16, 10, 16]  -> [None, 4, 5, 32]
        conv2 = tf.layers.max_pooling2d(conv2, 4, 2)


        conv3 = tf.layers.conv2d(conv2, 168, 3, padding='same',activation=None)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)
        # [None, 4, 5, 16]  -> [None, 1, 1, 168]
        conv3 = tf.layers.max_pooling2d(conv3, 4, 5)

        # squueze [None, 1, 1, 168] -> [None, 1,  168]
        conv3 = tf.squeeze(conv3, axis = 1)
        # [None, 1,  168] - > [None,168, 1] - > [None, 168, dim_1d]
        conv3_trans = tf.transpose(conv3, perm=[0,2,1])

        # [None,168, 1]  -> [None,   168, dim_1d]
        conv4 = tf.layers.conv1d(conv3_trans, dim_1d, 3, padding='same',activation=None)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training)
        conv4 = tf.nn.leaky_relu(conv4, alpha=0.2)

        # [None, 168, dim_1d]
        return conv4


    # take a list of feature maps, combine them through stacking
    # continue to train the stacked feature maps using several conv layers.
    # output a latent feature with specified dim
    def fuse_and_train(self, feature_map_list, is_training, suffix = '', dim=3):
        var_scope = 'fusion_layer_'+ suffix
        with tf.variable_scope(var_scope):
            fuse_feature =tf.concat(axis=3,values=feature_map_list)
            print('fuse_feature.shape: ', fuse_feature.shape)
            # Convolution Layer with 32 filters and a kernel size of 5
            # conv1 = tf.layers.conv2d(fuse_feature, 32, 3, padding='same',activation=my_leaky_relu)
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(fuse_feature, 16, 3, padding='same',activation=None)
            # conv1 = tf.layers.conv2d(x_2d_train_data, 16, 3, padding='same',activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            #  Convolution Layer with 64 filters and a kernel size of 3
            # conv2: change from 16 to 32
            conv2 = tf.layers.conv2d(conv1, 32, 3, padding='same',activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)


        # with tf.name_scope("fusion_batch_norm"):
        #     cnn2d_bn = tf.layers.batch_normalization(inputs=conv1, training=is_training)
        #     # (?, 168, 32, 20, 1)
        #     print('cnn2d_bn shape: ',cnn2d_bn.shape)

        # output should be (?, 32, 20, 1)
        # with tf.name_scope("fusion_layer_b"):
            conv3 = tf.layers.conv2d(
                      inputs=conv2,
                      filters=dim,
                      kernel_size=[1, 1],
                      padding="same",
                      activation=my_leaky_relu
                      #reuse = tf.AUTO_REUSE
                )

            out = conv3
            print('latent representation shape: ',out.shape)
        # output size should be [batchsize, height, width, dim]
        return out


    # take a latent fea, decode into [batchsize, 32, 20, dim_decode]
    def branching(self, latent_fea, dim_decode, is_training):
        padding = 'SAME'
        stride = [1, 1]
        #  [None, 32, 20, dim] - > [None, 32, 20, 16]
        conv1 = tf.layers.conv2d(latent_fea, 16, 3, padding='same',activation=None)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

        conv2 = tf.layers.conv2d(conv1, 32, 3, padding='same',activation=None)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)
        # [None, 32, 20, 16]  -> [None,32, 20 32]

        conv3 = tf.layers.conv2d(conv2, dim_decode, 3, padding='same',activation=None)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)
        #[None,32, 20 32] -> [None, 32, 20, dim_2d]
        return conv3


    '''
    inputs_: feature tensor: input shape: [None, timestep, height, width, channels]
             e.g. [None, 168, 32, 20, 9]
    to get the latent representation, obtain: encoded [None, 1, 32, 20, dims = 1]
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

            encoded = tf.layers.conv3d(inputs=maxpool4, filters= self.dim, kernel_size=[3,3,3], padding='same', activation=my_leaky_relu)
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



    '''
    TODO: output encoded layers for further grouping
    train_from_start: weather to train from scratch or not. If False, train
        from a pretrained all-to-all AE with part of the variables.

    pretrained_ckpt_path: pretrained model that provides intiialization of
        weights

    '''
    def train_autoencoder(self, rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict, train_hours,
                     demo_mask_arr, save_folder_path, dim, grouping_dict,
                     resume_training = False, checkpoint_path = None,
                     use_pretrained = False, pretrained_ckpt_path = None,
                       epochs=1, batch_size=32):
        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                       5000, 0.96, staircase=True)
        # first level output [dataset name: output]
        first_level_output = dict()
        for k, v in self.rawdata_1d_tf_x_dict.items():
            prediction_1d = self.cnn_1d_model(v, self.is_training, k)
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            prediction_1d_expand = tf.tile(prediction_1d, [1, HEIGHT,
                                                    WIDTH ,1])
            first_level_output[k] = prediction_1d_expand

        for k, v in self.rawdata_2d_tf_x_dict.items():
            prediction_2d = self.cnn_2d_model(v, self.is_training, k)
            first_level_output[k] = prediction_2d

        # for k, v in self.rawdata_3d_tf_x_dict.items():
        #     prediction_3d = self.cnn_model(v, self.is_training, k)
        #     first_level_output[k] = prediction_3d



        # ------------ grouping in encoder ------------- #
        # [group name: feature maps]
        second_level_output = dict()
        second_order_encoder_list = []  # output feature maps for grouping
        # second level key list, a list of group names to be used for further grouping
        keys_list = []

        for grp, data_list in grouping_dict.items():
            # group a list of dataset in a group
            temp_list = [] # a list of feature maps belonging to the same group from first level training
            for ds in data_list:
                temp_list.append(first_level_output[ds])

            scope_name = '1_'+ grp
            group_fusion_featuremap = self.fuse_and_train(temp_list, self.is_training, scope_name, dim=3) # fuse and train
            second_level_output[grp] = group_fusion_featuremap

            second_order_encoder_list.append(group_fusion_featuremap)
            keys_list.append(grp)


        # ------------------------------------------------#
        # dim: latent fea dimension
        latent_fea = self.fuse_and_train(list(second_level_output.values()),  self.is_training, '2', dim)
        print('latent_fea.shape: ', latent_fea.shape) # (?, 32, 20, 5)
        # recontruction
        print('recontruction')
        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
                # [1, 32, 20, 1]  -> [1, 1, 32, 20, 1]
                # [1, 32, 20, 1] -> [batchsize, 32, 20, 1]
                # batchsize = tf.shape(prediction)[0]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(latent_fea)[0],1,1,1])
        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)

        # ------------------ branching -----------------------------#
        # branch one latent feature into [# of groups]'s latent representations
        first_level_decode = dict()  # [group name: latent rep]
        for grp in list(grouping_dict.keys()):
            first_level_decode[grp] = self.branching(latent_fea, dim, self.is_training)

        # reconstruct all datasets
        # assumption: all datasets with equal weights
        total_loss = 0
    #    loss_dict = []  # {dataset name: loss}
        loss_dict = {}
        rmse_dict = {}
        # decode by groups
        keys_1d = rawdata_1d_dict.keys()
        keys_2d = rawdata_2d_dict.keys()
        keys_3d = rawdata_3d_dict.keys()
        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr_expanded, 1)

        for grp, data_list in grouping_dict.items():
            for ds in data_list:
                # reconstruct each
                if ds in keys_1d:
                    dim_1d = rawdata_1d_dict[ds].shape[-1]
                    reconstruction_1d = self.reconstruct_1d(first_level_decode[grp], dim_1d, self.is_training)
                    temp_loss = tf.losses.absolute_difference(reconstruction_1d, self.rawdata_1d_tf_y_dict[ds])
                    total_loss += temp_loss
                    loss_dict[ds] = temp_loss

                    temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_1d, self.rawdata_1d_tf_y_dict[ds]))
                    rmse_dict[ds] = temp_rmse

                if ds in keys_2d:
                    dim_2d = rawdata_2d_dict[ds].shape[-1]
                    reconstruction_2d = self.reconstruct_2d(first_level_decode[grp], dim_2d, self.is_training)
                    temp_loss = tf.losses.absolute_difference(reconstruction_2d, self.rawdata_2d_tf_y_dict[ds])
                    total_loss += temp_loss
                    loss_dict[ds] = temp_loss
                    temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_2d, self.rawdata_2d_tf_y_dict[ds]))
                    rmse_dict[ds] = temp_rmse

            #     if ds in keys_3d:
            #         timestep_3d = self.rawdata_3d_tf_y_dict[ds].shape[1]
            #         reconstruction_3d = self.reconstruct_3d(first_level_decode[grp], timestep_3d)
            # #         print('reconstruction_3d.shape: ', reconstruction_3d.shape) # (?, 7, 32, 20, 1)
            #         # 3d weight: (?, 32, 20, 1) -> (?, 7, 32, 20, 1)
            #         demo_mask_arr_temp = tf.tile(demo_mask_arr_expanded, [1, timestep_3d,1,1,1])
            #         weight_3d = tf.cast(tf.greater(demo_mask_arr_temp, 0), tf.float32)
            #         temp_loss = tf.losses.absolute_difference(reconstruction_3d, self.rawdata_3d_tf_y_dict[ds], weight_3d)
            #         total_loss += temp_loss
            #         loss_dict[ds] = temp_loss
            #         temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_3d, self.rawdata_3d_tf_y_dict[ds]))
            #         rmse_dict[ds] = temp_rmse


        print('total_loss: ', total_loss)
        cost = total_loss

        #--------  fix weights, not update during optimization ------ #
        variables = tf.global_variables()
        # get scopes_to_reserve
        scopes_to_reserve = get_scopes_to_restore(rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict)
        variable_to_restore = get_variables_to_restore(variables, scopes_to_reserve)
        print('variable_to_restore: ')
        print(variable_to_restore)
        variables_to_update = [v for v in tf.global_variables() if v not in variable_to_restore]

        with tf.name_scope("training"):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost,
                    global_step = self.global_step)


        train_result = list()
        test_result = list()
        encoded_list = list()  # output last layer of encoded for further grouping

        if not os.path.exists(save_folder_path):
            os.makedirs(save_path)


        # --- dealing with model saver ------ #
        if not use_pretrained:
            saver = tf.train.Saver()
        else:
            print('Restoring saver from pretrained model....', pretrained_ckpt_path)
            # train from pretrained model_fusion
            vars_to_restore_dict = {}
            # make the dictionary, note that everything here will have “:0”, avoid it.
            for v in variable_to_restore:
                vars_to_restore_dict[v.name[:-2]] = v
            # only for restoring pretrained model weights
            pretrained_saver = tf.train.Saver(vars_to_restore_dict)
             # save all variables
            saver = tf.train.Saver()



        ########### start session ########################
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # ----- if initialized with pretrained weights ----
            if use_pretrained:
                pretrained_saver.restore(sess, pretrained_ckpt_path)

            # ---- if resume training -----
            if resume_training:
                if checkpoint_path is not None:
                    saver.restore(sess, checkpoint_path)
                else:
                    saver.restore(sess, tf.train.latest_checkpoint(save_folder_path))
                # check global step
                print("global step: ", sess.run([self.global_step]))
                print("Model restore finished, current globle step: %d" % self.global_step.eval())

                # get new epoch num
                print("int(train_hours / batch_size +1): ", int(train_hours / batch_size +1))
                start_epoch_num = tf.div(self.global_step, int(train_hours / batch_size +1))
                #self.global_step/ (len(x_train_data) / batch_size +1) -1
                print("start_epoch_num: ", start_epoch_num.eval())
                start_epoch = start_epoch_num.eval()
            else:
                start_epoch = 0

            # temporary
            # train_hours = 200
            # train_hours: train_start_time = '2014-02-01',train_end_time = '2018-10-31',
            if train_hours%batch_size ==0:
                iterations = int(train_hours/batch_size)
            else:
                iterations = int(train_hours/batch_size) + 1

            for epoch in range(start_epoch, epochs):
                print('Epoch', epoch, 'started', end='')
                start_time = datetime.datetime.now()
                epoch_loss = 0
                epoch_subloss = {}  # ave loss for each dataset
                epoch_subloss = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

                epoch_subrmse = {}  # ave loss for each dataset
                epoch_subrmse = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

                final_output = list()
                final_encoded_list = list()

                # mini batch
                for itr in range(iterations):
                    start_idx = itr*batch_size
                    if train_hours < (itr+1)*batch_size:
                        end_idx = train_hours
                    else:
                        end_idx = (itr+1)*batch_size
                    print('itr, start_idx, end_idx', itr, start_idx, end_idx)

                    # create feed_dict
                    feed_dict_all = {}  # tf_var:  tensor
                    # create batches for 1d
                    for k, v in rawdata_1d_dict.items():
                        temp_batch = create_mini_batch_1d(start_idx, end_idx, v)
                        feed_dict_all[self.rawdata_1d_tf_x_dict[k]] = temp_batch
                        feed_dict_all[self.rawdata_1d_tf_y_dict[k]] = temp_batch

                    # create batches for 2d
                    for k, v in rawdata_2d_dict.items():
                        temp_batch = create_mini_batch_2d(start_idx, end_idx, v)
                        feed_dict_all[self.rawdata_2d_tf_x_dict[k]] = temp_batch
                        feed_dict_all[self.rawdata_2d_tf_y_dict[k]] = temp_batch

                     # create batches for 3d
    #                 for k, v in rawdata_3d_dict.items():
    #                     if k == 'seattle911calls':
    #                         timestep = 168
    #                     else:
    #                         timestep = 7
    #                     temp_batch = create_mini_batch_3d(start_idx, end_idx, v, timestep)
    # #                     print('3d temp_batch.shape: ',temp_batch.shape)
    #                     feed_dict_all[self.rawdata_3d_tf_x_dict[k]] = temp_batch
    #                     feed_dict_all[self.rawdata_3d_tf_y_dict[k]] = temp_batch
                    # is_training: True
                    feed_dict_all[self.is_training] = True
                    batch_cost, batch_loss_dict, batch_rmse_dict, _ = sess.run([cost,loss_dict, rmse_dict, optimizer], feed_dict=feed_dict_all)
                    # get encoded representation
                    # # [None, 1, 32, 20, 1]
                    batch_output, batch_encoded_list = sess.run([latent_fea, second_order_encoder_list], feed_dict= feed_dict_all)
                    final_output.extend(batch_output)

                    # record results every 50 iterations, that is about 900 samples
                    if itr% 50 == 0:
                        final_encoded_list.append(batch_encoded_list)

                    epoch_loss += batch_cost
                    for k, v in epoch_subloss.items():
                        epoch_subloss[k] += batch_loss_dict[k]

                    for k, v in epoch_subrmse.items():
                        epoch_subrmse[k] += batch_rmse_dict[k]


                    if itr%10 == 0:
                        print("Iter/Epoch: {}/{}...".format(itr, epoch),
                            "Training loss: {:.4f}".format(batch_cost))
                        for k, v in batch_loss_dict.items():
                            print('loss for k :', k, v)


                # report loss per epoch
                epoch_loss = epoch_loss/ iterations
                print('epoch: ', epoch, 'Trainig Set Epoch total Cost: ',epoch_loss)
                end_time = datetime.datetime.now()
                train_time_per_epoch = end_time - start_time
                train_time_per_sample = train_time_per_epoch/ train_hours

                print(' Training Time per epoch: ', str(train_time_per_epoch), 'Time per sample: ', str(train_time_per_sample))

                for k, v in epoch_subloss.items():
                    epoch_subloss[k] = v/iterations
                    print('epoch: ', epoch, 'k: ', k, 'mean train loss: ', epoch_subloss[k])

                for k, v in epoch_subrmse.items():
                    epoch_subrmse[k] = v/iterations
                    print('epoch: ', epoch, 'k: ', k, 'mean train rmse: ', epoch_subrmse[k])


                save_path = saver.save(sess, save_folder_path +'autoencoder_v6_' +str(epoch)+'.ckpt', global_step=self.global_step)
                # save_path = saver.save(sess, './autoencoder.ckpt')
                print('Model saved to {}'.format(save_path))

                # Testing per epoch
                # -----------------------------------------------------------------
                print('testing per epoch, for epoch: ', epoch)
                # train_hours  = 41616  # train_start_time = '2014-02-01',train_end_time = '2018-10-31'
                test_start = train_hours
                test_end = rawdata_1d_dict['weather'].shape[0] -168  # 45984 - 168
                test_len = test_end - test_start  # 4200
                print('test_start: ', test_start) # 41616
                print('test_end: ', test_end)
                print('test_len: ', test_len) #  4200
                test_start_time = datetime.datetime.now()

                test_cost = 0
                test_final_output = list()
                test_subloss = {}  # ave loss for each dataset
                test_subloss = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

                test_subrmse = {}  # ave loss for each dataset
                test_subrmse = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))


                if test_len%batch_size ==0:
                    itrs = int(test_len/batch_size)
                else:
                    itrs = int(test_len/batch_size) + 1

                for itr in range(itrs):
                    start_idx = itr*batch_size + test_start
                    if test_len < (itr+1)*batch_size:
                        end_idx = test_end
                    else:
                        end_idx = (itr+1)*batch_size + test_start
                    print('testing: start_idx, end_idx', start_idx, end_idx)
                    # create feed_dict
                    test_feed_dict_all = {}  # tf_var:  tensor
                    # create batches for 1d
                    for k, v in rawdata_1d_dict.items():
                        temp_batch = create_mini_batch_1d(start_idx, end_idx, v)
                        test_feed_dict_all[self.rawdata_1d_tf_x_dict[k]] = temp_batch
                        test_feed_dict_all[self.rawdata_1d_tf_y_dict[k]] = temp_batch

                    # create batches for 2d
                    for k, v in rawdata_2d_dict.items():
                        temp_batch = create_mini_batch_2d(start_idx, end_idx, v)
                        test_feed_dict_all[self.rawdata_2d_tf_x_dict[k]] = temp_batch
                        test_feed_dict_all[self.rawdata_2d_tf_y_dict[k]] = temp_batch

                     # create batches for 3d
    #                 for k, v in rawdata_3d_dict.items():
    #                     if k == 'seattle911calls':
    #                         timestep = 168
    #                     else:
    #                         timestep = 7
    #                     temp_batch = create_mini_batch_3d(start_idx, end_idx, v, timestep)
    # #                     print('3d temp_batch.shape: ',temp_batch.shape)
    #                     test_feed_dict_all[self.rawdata_3d_tf_x_dict[k]] = temp_batch
    #                     test_feed_dict_all[self.rawdata_3d_tf_y_dict[k]] = temp_batch


                    # is_training: True
                    test_feed_dict_all[self.is_training] = True

                    test_batch_cost, test_batch_loss_dict, test_batch_rmse_dict = sess.run([cost,loss_dict, rmse_dict], feed_dict= test_feed_dict_all)
                    # get encoded representation
                    # # [None, 1, 32, 20, 1]
                    test_batch_output = sess.run([latent_fea], feed_dict= test_feed_dict_all)
                    test_final_output.extend(test_batch_output)

                    for k, v in test_subloss.items():
                        test_subloss[k] += test_batch_loss_dict[k]

                    for k, v in test_subrmse.items():
                        test_subrmse[k] += test_batch_rmse_dict[k]


                    if itr%10 == 0:
                        print("Iter/Epoch: {}/{}...".format(itr, epoch),
                            "testing loss: {:.4f}".format(test_batch_cost))


                    # test_mini_batch_x = self.create_mini_batch(start_idx, end_idx, data_1d, data_2d, data_3d)
                    #
                    # test_batch_cost, _ = sess.run([cost, optimizer], feed_dict={self.x: test_mini_batch_x,
                    #                                                 self.y: test_mini_batch_x})
                    # get encoded representation
                    # # [None, 1, 32, 20, 1]
                    # test_batch_output = sess.run([encoded], feed_dict={self.x: test_mini_batch_x,
                    #                                                 self.y: test_mini_batch_x})

                    test_cost += test_batch_cost

                test_epoch_loss = test_cost/ itrs
                print('epoch: ', epoch, 'Test Set Epoch total Cost: ',test_epoch_loss)
                test_end_time = datetime.datetime.now()
                test_time_per_epoch = test_end_time - test_start_time
                test_time_per_sample = test_time_per_epoch/ test_len
                print(' test Time elapse: ', str(test_time_per_epoch), 'test Time per sample: ', str(test_time_per_sample))

                for k, v in test_subloss.items():
                    test_subloss[k] = v/itrs
                    print('epoch: ', epoch, 'k: ', k, 'mean test loss: ', test_subloss[k])
                    print('test loss for k :', k, v)

                for k, v in test_subrmse.items():
                    test_subrmse[k] = v/itrs
                    print('epoch: ', epoch, 'k: ', k, 'mean test rmse: ', test_subrmse[k])
                    print('test rmse for k :', k, v)


                # -----------------------------------------------------------------------


                # save epoch statistics to csv
                ecoch_res_df = pd.DataFrame([[epoch_loss, test_epoch_loss]],
                    columns=[ 'train_loss', 'test_loss'])
                res_csv_path = save_folder_path + 'autoencoder_ecoch_res_df' +'.csv'
                with open(res_csv_path, 'a') as f:
                    # Add header if file is being created, otherwise skip it
                    ecoch_res_df.to_csv(f, header=f.tell()==0)


                train_sub_res_df = pd.DataFrame([list(epoch_subloss.values())],
                    columns= list(epoch_subloss.keys()))
                train_sub_res_csv_path = save_folder_path + 'autoencoder_train_sub_res' +'.csv'
                with open(train_sub_res_csv_path, 'a') as f:
                    train_sub_res_df.to_csv(f, header=f.tell()==0)



                test_sub_res_df = pd.DataFrame([list(test_subloss.values())],
                                columns= list(test_subloss.keys()))
                test_sub_res_csv_path = save_folder_path + 'autoencoder_test_sub_res' +'.csv'
                with open(test_sub_res_csv_path, 'a') as f:
                    test_sub_res_df.to_csv(f, header=f.tell()==0)


                # --- rmse ------
                train_sub_rmse_df = pd.DataFrame([list(epoch_subrmse.values())],
                    columns= list(epoch_subrmse.keys()))
                train_sub_rmse_csv_path = save_folder_path + 'autoencoder_train_sub_rmse' +'.csv'
                with open(train_sub_rmse_csv_path, 'a') as f:
                    train_sub_rmse_df.to_csv(f, header=f.tell()==0)



                test_sub_rmse_df = pd.DataFrame([list(test_subrmse.values())],
                                columns= list(test_subrmse.keys()))
                test_sub_rmse_csv_path = save_folder_path + 'autoencoder_test_sub_rmse' +'.csv'
                with open(test_sub_rmse_csv_path, 'a') as f:
                    test_sub_rmse_df.to_csv(f, header=f.tell()==0)



                # save results to txt
                txt_name = save_folder_path + 'AE_v5_df' +  '.txt'
                with open(txt_name, 'w') as the_file:
                    #the_file.write('Only account for grids that intersect with city boundary \n')
                    the_file.write('epoch\n')
                    the_file.write(str(epoch)+'\n')
                    the_file.write('dim\n')
                    the_file.write(str(self.dim) + '\n')
                    the_file.write(' epoch_loss:\n')
                    the_file.write(str(epoch_loss) + '\n')
                    the_file.write(' test_epoch_loss:\n')
                    the_file.write(str(test_epoch_loss) + '\n')
                    the_file.write('\n')
                    the_file.write('total time of last train epoch\n')
                    the_file.write(str(train_time_per_epoch) + '\n')
                    the_file.write('time per sample for train\n')
                    the_file.write(str(train_time_per_sample) + '\n')
                    the_file.write('total time of last test epoch\n')
                    the_file.write(str(test_time_per_epoch) + '\n')
                    the_file.write('time per sample for test\n')
                    the_file.write(str(test_time_per_sample) + '\n')
                    the_file.write('keys_list\n')
                    for item in keys_list:
                        the_file.write("%s\n" % item)
                    the_file.close()

                # plot results
                print('saving train_test plots')
                train_test = pd.read_csv(save_folder_path  + 'autoencoder_ecoch_res_df' +'.csv')
                # train_test = train_test.loc[:, ~train_test.columns.str.contains('^Unnamed')]
                train_test[['train_loss', 'test_loss']].plot()
                plt.savefig(save_folder_path + 'total_loss_inprogress.png')
                # train_test[['train_acc', 'test_acc']].plot()
                # plt.savefig(save_folder_path + 'acc_loss_inprogress.png')

                plt.close()

                if epoch == epochs-1:
                    final_output = np.array(final_output)
                    train_result.extend(final_output)
                    test_final_output = np.array(test_final_output)
                    test_result.extend(test_final_output)
                    encoded_list.extend(final_encoded_list)

            # encoded_res = np.array(test_result)
            train_encoded_res = train_result
            train_output_arr = train_encoded_res[0]
            for i in range(1,len(train_encoded_res)):
                train_output_arr = np.concatenate((train_output_arr, train_encoded_res[i]), axis=0)

            test_encoded_res = test_result
            test_output_arr = test_encoded_res[0]
            for i in range(1,len(test_encoded_res)):
                test_output_arr = np.concatenate((test_output_arr, test_encoded_res[i]), axis=0)

        # This is the latent representation (9337, 1, 32, 20, 1) of training
        return train_output_arr, test_output_arr, encoded_list, keys_list


    # output all intermediate latent representations in encoding part
    def inference_autoencoder(self, rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict, train_hours,
                     demo_mask_arr, save_folder_path, dim, grouping_dict,
                    checkpoint_path = None,
                       epochs=1, batch_size=32):
                # first level output [dataset name: output]
        first_level_output = dict()
        for k, v in self.rawdata_1d_tf_x_dict.items():
            prediction_1d = self.cnn_1d_model(v, self.is_training, k)
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            prediction_1d = tf.expand_dims(prediction_1d, 1)
            prediction_1d_expand = tf.tile(prediction_1d, [1, HEIGHT,
                                                    WIDTH ,1])
            first_level_output[k] = prediction_1d_expand

        for k, v in self.rawdata_2d_tf_x_dict.items():
            prediction_2d = self.cnn_2d_model(v, self.is_training, k)
            first_level_output[k] = prediction_2d

        # for k, v in self.rawdata_3d_tf_x_dict.items():
        #     prediction_3d = self.cnn_model(v, self.is_training, k)
        #     first_level_output[k] = prediction_3d

        # ------------ grouping in encoder ------------- #
        # [group name: feature maps]
        second_level_output = dict()
        second_order_encoder_list = []  # output feature maps for grouping
        # second level key list, a list of group names to be used for further grouping
        keys_list = []

        for grp, data_list in grouping_dict.items():
            # group a list of dataset in a group
            temp_list = [] # a list of feature maps belonging to the same group from first level training
            for ds in data_list:
                temp_list.append(first_level_output[ds])

            scope_name = '1_'+ grp
            group_fusion_featuremap = self.fuse_and_train(temp_list, self.is_training, scope_name, dim=3) # fuse and train
            second_level_output[grp] = group_fusion_featuremap

            second_order_encoder_list.append(group_fusion_featuremap)
            keys_list.append(grp)


        # ------------------------------------------------#
        # dim: latent fea dimension
        latent_fea = self.fuse_and_train(list(second_level_output.values()),  self.is_training, '2', dim)
        print('latent_fea.shape: ', latent_fea.shape) # (?, 32, 20, 5)
        # recontruction
        print('recontruction')
        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
                # [1, 32, 20, 1]  -> [1, 1, 32, 20, 1]
                # [1, 32, 20, 1] -> [batchsize, 32, 20, 1]
                # batchsize = tf.shape(prediction)[0]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(latent_fea)[0],1,1,1])
        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)

        # ------------------ branching -----------------------------#
        # branch one latent feature into [# of groups]'s latent representations
        first_level_decode = dict()  # [group name: latent rep]
        for grp in list(grouping_dict.keys()):
            first_level_decode[grp] = self.branching(latent_fea, dim, self.is_training)

        # reconstruct all datasets
        # assumption: all datasets with equal weights
        total_loss = 0
    #    loss_dict = []  # {dataset name: loss}
        loss_dict = {}
        rmse_dict = {}
        # decode by groups
        keys_1d = rawdata_1d_dict.keys()
        keys_2d = rawdata_2d_dict.keys()
        keys_3d = rawdata_3d_dict.keys()
        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr_expanded, 1)

        for grp, data_list in grouping_dict.items():
            for ds in data_list:
                # reconstruct each
                if ds in keys_1d:
                    dim_1d = rawdata_1d_dict[ds].shape[-1]
                    reconstruction_1d = self.reconstruct_1d(first_level_decode[grp], dim_1d, self.is_training)
                    temp_loss = tf.losses.absolute_difference(reconstruction_1d, self.rawdata_1d_tf_y_dict[ds])
                    total_loss += temp_loss
                    loss_dict[ds] = temp_loss

                    temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_1d, self.rawdata_1d_tf_y_dict[ds]))
                    rmse_dict[ds] = temp_rmse

                if ds in keys_2d:
                    dim_2d = rawdata_2d_dict[ds].shape[-1]
                    reconstruction_2d = self.reconstruct_2d(first_level_decode[grp], dim_2d, self.is_training)
                    temp_loss = tf.losses.absolute_difference(reconstruction_2d, self.rawdata_2d_tf_y_dict[ds])
                    total_loss += temp_loss
                    loss_dict[ds] = temp_loss
                    temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_2d, self.rawdata_2d_tf_y_dict[ds]))
                    rmse_dict[ds] = temp_rmse

            #     if ds in keys_3d:
            #         timestep_3d = self.rawdata_3d_tf_y_dict[ds].shape[1]
            #         reconstruction_3d = self.reconstruct_3d(first_level_decode[grp], timestep_3d)
            # #         print('reconstruction_3d.shape: ', reconstruction_3d.shape) # (?, 7, 32, 20, 1)
            #         # 3d weight: (?, 32, 20, 1) -> (?, 7, 32, 20, 1)
            #         demo_mask_arr_temp = tf.tile(demo_mask_arr_expanded, [1, timestep_3d,1,1,1])
            #         weight_3d = tf.cast(tf.greater(demo_mask_arr_temp, 0), tf.float32)
            #         temp_loss = tf.losses.absolute_difference(reconstruction_3d, self.rawdata_3d_tf_y_dict[ds], weight_3d)
            #         total_loss += temp_loss
            #         loss_dict[ds] = temp_loss
            #         temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_3d, self.rawdata_3d_tf_y_dict[ds]))
            #         rmse_dict[ds] = temp_rmse


        print('total_loss: ', total_loss)
        cost = total_loss


        train_result = list()
        test_result = list()
        encoded_list = list()  # output last layer of encoded for further grouping
        test_encoded_list = list()

        if not os.path.exists(save_folder_path):
            os.makedirs(save_path)

        save_folder_path = os.path.join(save_folder_path, 'inference/')

        saver = tf.train.Saver()

        ########### start session ########################
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # ---- if resume training -----
            if checkpoint_path is not None:
                saver.restore(sess, checkpoint_path)
            else:
                saver.restore(sess, tf.train.latest_checkpoint(save_folder_path))
                # check global step

            # temporary
            # train_hours = 200
            # train_hours: train_start_time = '2014-02-01',train_end_time = '2018-10-31',
            if train_hours%batch_size ==0:
                iterations = int(train_hours/batch_size)
            else:
                iterations = int(train_hours/batch_size) + 1

            start_time = datetime.datetime.now()
            epoch_loss = 0
            epoch_subloss = {}  # ave loss for each dataset
            epoch_subloss = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

            epoch_subrmse = {}  # ave loss for each dataset
            epoch_subrmse = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

            final_output = list()
            final_encoded_list = list()

            # mini batch
            for itr in range(iterations):
                start_idx = itr*batch_size
                if train_hours < (itr+1)*batch_size:
                    end_idx = train_hours
                else:
                    end_idx = (itr+1)*batch_size
                print('itr, start_idx, end_idx', itr, start_idx, end_idx)

                # create feed_dict
                feed_dict_all = {}  # tf_var:  tensor
                    # create batches for 1d
                for k, v in rawdata_1d_dict.items():
                    temp_batch = create_mini_batch_1d(start_idx, end_idx, v)
                    feed_dict_all[self.rawdata_1d_tf_x_dict[k]] = temp_batch
                    feed_dict_all[self.rawdata_1d_tf_y_dict[k]] = temp_batch

                    # create batches for 2d
                for k, v in rawdata_2d_dict.items():
                    temp_batch = create_mini_batch_2d(start_idx, end_idx, v)
                    feed_dict_all[self.rawdata_2d_tf_x_dict[k]] = temp_batch
                    feed_dict_all[self.rawdata_2d_tf_y_dict[k]] = temp_batch

                     # create batches for 3d
    #                 for k, v in rawdata_3d_dict.items():
    #                     if k == 'seattle911calls':
    #                         timestep = 168
    #                     else:
    #                         timestep = 7
    #                     temp_batch = create_mini_batch_3d(start_idx, end_idx, v, timestep)
    # #                     print('3d temp_batch.shape: ',temp_batch.shape)
    #                     feed_dict_all[self.rawdata_3d_tf_x_dict[k]] = temp_batch
    #                     feed_dict_all[self.rawdata_3d_tf_y_dict[k]] = temp_batch
                    # is_training: True
                feed_dict_all[self.is_training] = True
                batch_cost, batch_loss_dict, batch_rmse_dict = sess.run([cost,loss_dict, rmse_dict], feed_dict=feed_dict_all)
                    # get encoded representation
                    # # [None, 1, 32, 20, 1]
                batch_output, batch_encoded_list = sess.run([latent_fea, second_order_encoder_list], feed_dict= feed_dict_all)
                final_output.extend(batch_output)

                final_encoded_list.append(batch_encoded_list)

                epoch_loss += batch_cost
                for k, v in epoch_subloss.items():
                    epoch_subloss[k] += batch_loss_dict[k]

                for k, v in epoch_subrmse.items():
                    epoch_subrmse[k] += batch_rmse_dict[k]


                if itr%10 == 0:
                    print("Iter: {}...".format(itr),
                            "Training loss: {:.4f}".format(batch_cost))
                    for k, v in batch_loss_dict.items():
                        print('loss for k :', k, v)


            # report loss per epoch
            epoch_loss = epoch_loss/ iterations
            print('Trainig Set Epoch total Cost: ',epoch_loss)
            end_time = datetime.datetime.now()
            train_time_per_epoch = end_time - start_time
            train_time_per_sample = train_time_per_epoch/ train_hours

            print(' Training Time per epoch: ', str(train_time_per_epoch), 'Time per sample: ', str(train_time_per_sample))

            for k, v in epoch_subloss.items():
                epoch_subloss[k] = v/iterations
                # print('epoch: ', epoch, 'k: ', k, 'mean train loss: ', epoch_subloss[k])

            for k, v in epoch_subrmse.items():
                epoch_subrmse[k] = v/iterations
                # print('epoch: ', epoch, 'k: ', k, 'mean train rmse: ', epoch_subrmse[k])


            save_path = saver.save(sess, save_folder_path +'infer_autoencoder_v6_' +'.ckpt', global_step=self.global_step)
                # save_path = saver.save(sess, './autoencoder.ckpt')
            print('Model saved to {}'.format(save_path))

                # Testing per epoch
                # -----------------------------------------------------------------
            print('testing  ')
                # train_hours  = 41616  # train_start_time = '2014-02-01',train_end_time = '2018-10-31'
            test_start = train_hours
            test_end = rawdata_1d_dict['weather'].shape[0] -168  # 45984 - 168
            test_len = test_end - test_start  # 4200
            print('test_start: ', test_start) # 41616
            print('test_end: ', test_end)
            print('test_len: ', test_len) #  4200
            test_start_time = datetime.datetime.now()

            test_cost = 0
            test_final_output = list()
            test_final_encoded_list = list()
            test_subloss = {}  # ave loss for each dataset
            test_subloss = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

            test_subrmse = {}  # ave loss for each dataset
            test_subrmse = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))


            if test_len%batch_size ==0:
                itrs = int(test_len/batch_size)
            else:
                itrs = int(test_len/batch_size) + 1

            for itr in range(itrs):
                start_idx = itr*batch_size + test_start
                if test_len < (itr+1)*batch_size:
                    end_idx = test_end
                else:
                    end_idx = (itr+1)*batch_size + test_start
                print('testing: start_idx, end_idx', start_idx, end_idx)
                    # create feed_dict
                test_feed_dict_all = {}  # tf_var:  tensor
                    # create batches for 1d
                for k, v in rawdata_1d_dict.items():
                    temp_batch = create_mini_batch_1d(start_idx, end_idx, v)
                    test_feed_dict_all[self.rawdata_1d_tf_x_dict[k]] = temp_batch
                    test_feed_dict_all[self.rawdata_1d_tf_y_dict[k]] = temp_batch

                    # create batches for 2d
                for k, v in rawdata_2d_dict.items():
                    temp_batch = create_mini_batch_2d(start_idx, end_idx, v)
                    test_feed_dict_all[self.rawdata_2d_tf_x_dict[k]] = temp_batch
                    test_feed_dict_all[self.rawdata_2d_tf_y_dict[k]] = temp_batch

                     # create batches for 3d
    #                 for k, v in rawdata_3d_dict.items():
    #                     if k == 'seattle911calls':
    #                         timestep = 168
    #                     else:
    #                         timestep = 7
    #                     temp_batch = create_mini_batch_3d(start_idx, end_idx, v, timestep)
    # #                     print('3d temp_batch.shape: ',temp_batch.shape)
    #                     test_feed_dict_all[self.rawdata_3d_tf_x_dict[k]] = temp_batch
    #                     test_feed_dict_all[self.rawdata_3d_tf_y_dict[k]] = temp_batch


                    # is_training: True
                test_feed_dict_all[self.is_training] = True

                test_batch_cost, test_batch_loss_dict, test_batch_rmse_dict = sess.run([cost,loss_dict, rmse_dict], feed_dict= test_feed_dict_all)
                    # get encoded representation
                    # # [None, 1, 32, 20, 1]
                test_batch_output, test_batch_encoded_list = sess.run([latent_fea, second_order_encoder_list], feed_dict= test_feed_dict_all)
                test_final_output.extend(test_batch_output)

                test_final_encoded_list.append(test_batch_encoded_list)

                for k, v in test_subloss.items():
                    test_subloss[k] += test_batch_loss_dict[k]

                for k, v in test_subrmse.items():
                    test_subrmse[k] += test_batch_rmse_dict[k]


                if itr%10 == 0:
                    print("Iter: {}...".format(itr),
                            "testing loss: {:.4f}".format(test_batch_cost))



                test_cost += test_batch_cost
########
            test_epoch_loss = test_cost/ itrs
            print('Test Set Epoch total Cost: ',test_epoch_loss)
            test_end_time = datetime.datetime.now()
            test_time_per_epoch = test_end_time - test_start_time
            test_time_per_sample = test_time_per_epoch/ test_len
            print(' test Time elapse: ', str(test_time_per_epoch), 'test Time per sample: ', str(test_time_per_sample))

            for k, v in test_subloss.items():
                test_subloss[k] = v/itrs
                # print('epoch: ', epoch, 'k: ', k, 'mean test loss: ', test_subloss[k])
                print('test loss for k :', k, v)

            for k, v in test_subrmse.items():
                test_subrmse[k] = v/itrs
                # print('epoch: ', epoch, 'k: ', k, 'mean test rmse: ', test_subrmse[k])
                print('test rmse for k :', k, v)


                # -----------------------------------------------------------------------


                # save epoch statistics to csv
            ecoch_res_df = pd.DataFrame([[epoch_loss, test_epoch_loss]],
                    columns=[ 'train_loss', 'test_loss'])
            res_csv_path = save_folder_path + 'autoencoder_ecoch_res_df' +'.csv'
            with open(res_csv_path, 'a') as f:
                    # Add header if file is being created, otherwise skip it
                ecoch_res_df.to_csv(f, header=f.tell()==0)


            train_sub_res_df = pd.DataFrame([list(epoch_subloss.values())],
                    columns= list(epoch_subloss.keys()))
            train_sub_res_csv_path = save_folder_path + 'autoencoder_train_sub_res' +'.csv'
            with open(train_sub_res_csv_path, 'a') as f:
                train_sub_res_df.to_csv(f, header=f.tell()==0)



            test_sub_res_df = pd.DataFrame([list(test_subloss.values())],
                                columns= list(test_subloss.keys()))
            test_sub_res_csv_path = save_folder_path + 'autoencoder_test_sub_res' +'.csv'
            with open(test_sub_res_csv_path, 'a') as f:
                test_sub_res_df.to_csv(f, header=f.tell()==0)


                # --- rmse ------
            train_sub_rmse_df = pd.DataFrame([list(epoch_subrmse.values())],
                    columns= list(epoch_subrmse.keys()))
            train_sub_rmse_csv_path = save_folder_path + 'autoencoder_train_sub_rmse' +'.csv'
            with open(train_sub_rmse_csv_path, 'a') as f:
                train_sub_rmse_df.to_csv(f, header=f.tell()==0)

            test_sub_rmse_df = pd.DataFrame([list(test_subrmse.values())],
                                columns= list(test_subrmse.keys()))
            test_sub_rmse_csv_path = save_folder_path + 'autoencoder_test_sub_rmse' +'.csv'
            with open(test_sub_rmse_csv_path, 'a') as f:
                test_sub_rmse_df.to_csv(f, header=f.tell()==0)



            # save results to txt
            txt_name = save_folder_path + 'infer_AE_v6_df' +  '.txt'
            with open(txt_name, 'w') as the_file:
                    #the_file.write('Only account for grids that intersect with city boundary \n')
                # the_file.write('epoch\n')
                # the_file.write(str(epoch)+'\n')
                the_file.write('dim\n')
                the_file.write(str(self.dim) + '\n')
                the_file.write(' epoch_loss:\n')
                the_file.write(str(epoch_loss) + '\n')
                the_file.write(' test_epoch_loss:\n')
                the_file.write(str(test_epoch_loss) + '\n')
                the_file.write('\n')
                the_file.write('total time of last train epoch\n')
                the_file.write(str(train_time_per_epoch) + '\n')
                the_file.write('time per sample for train\n')
                the_file.write(str(train_time_per_sample) + '\n')
                the_file.write('total time of last test epoch\n')
                the_file.write(str(test_time_per_epoch) + '\n')
                the_file.write('time per sample for test\n')
                the_file.write(str(test_time_per_sample) + '\n')
                the_file.write('keys_list\n')
                for item in keys_list:
                    the_file.write("%s\n" % item)
                the_file.close()


            final_output = np.array(final_output)
            train_result.extend(final_output)
            test_final_output = np.array(test_final_output)
            test_result.extend(test_final_output)
            encoded_list.extend(final_encoded_list)
            test_encoded_list.extend(test_final_encoded_list)



            print('saving output_arr ....')
            train_encoded_res = train_result
            train_output_arr = train_encoded_res[0]
            for i in range(1,len(train_encoded_res)):
                train_output_arr = np.concatenate((train_output_arr, train_encoded_res[i]), axis=0)

            test_encoded_res = test_result
            test_output_arr = test_encoded_res[0]
            for i in range(1,len(test_encoded_res)):
                test_output_arr = np.concatenate((test_output_arr, test_encoded_res[i]), axis=0)

        print('train_output_arr.shape: ', train_output_arr.shape)
        # This is the latent representation (9337, 1, 32, 20, 1) of training
        return train_output_arr, test_output_arr, encoded_list, test_encoded_list,keys_list




'''
fixed lenght time window: 168 hours
'''
class Autoencoder_entry:
    def __init__(self, train_obj,
              rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict, intersect_pos_set,
                    demo_mask_arr, save_path, dim, grouping_dict,
                    HEIGHT, WIDTH, TIMESTEPS, CHANNEL, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                     is_inference = False, checkpoint_path = None,
                     resume_training = False, train_dir = None,
                     use_pretrained = False, pretrained_ckpt_path = None,
                      # weather to train from pretrained models
                     ):
                     #  if s_inference = True, do inference only
        self.train_obj = train_obj
        self.train_hours = train_obj.train_hours
        self.rawdata_1d_dict = rawdata_1d_dict
        self.rawdata_2d_dict = rawdata_2d_dict
        self.rawdata_3d_dict = rawdata_3d_dict

        self.intersect_pos_set = intersect_pos_set
        self.demo_mask_arr = demo_mask_arr
        self.save_path = save_path
        self.dim = dim
        self.grouping_dict = grouping_dict

        globals()['HEIGHT']  = HEIGHT
        globals()['WIDTH']  = WIDTH
        globals()['TIMESTEPS']  = TIMESTEPS
        globals()['CHANNEL']  = CHANNEL
        globals()['BATCH_SIZE']  = BATCH_SIZE
        globals()['TRAINING_STEPS']  = TRAINING_STEPS
        globals()['LEARNING_RATE']  = LEARNING_RATE

        print('HEIGHT: ', HEIGHT)
        print('start learning rate: ',LEARNING_RATE)

        self.is_inference = is_inference
        self.checkpoint_path = checkpoint_path
        self.resume_training = resume_training
        self.train_dir = train_dir

        self.use_pretrained = use_pretrained
        self.pretrained_ckpt_path = pretrained_ckpt_path

        # ignore non-intersection cells in test_df
        # this is for evaluation
        # self.test_df_cut = self.test_df.loc[:,self.test_df.columns.isin(list(self.intersect_pos_set))]

        if is_inference == False:
            if resume_training == False:
                    # get prediction results
                    print('training from scratch, and get prediction results')
                    # predicted_vals: (552, 30, 30, 1)
                    self.train_lat_rep, self.test_lat_rep, encoded_list, keys_list = self.run_autoencoder()
                    # np.save(self.save_path +'train_lat_rep.npy', self.train_lat_rep)
                    # np.save(self.save_path +'test_lat_rep.npy', self.test_lat_rep)
            else:
                    # resume training
                    print("resume training, and get prediction results")
                    self.train_lat_rep, self.test_lat_rep, encoded_list, keys_list = self.run_resume_training()
            np.save(self.save_path +'train_lat_rep.npy', self.train_lat_rep)
            np.save(self.save_path +'test_lat_rep.npy', self.test_lat_rep)
            file = open(self.save_path + 'encoded_list', 'wb')
            # dump information to that file
            # number of batches, num_dataset, batchsize, h, w, dim
            print('dumping encoded_list to pickle')
            pickle.dump(encoded_list, file)
            file.close()
        else:
            # inference only
            # dumpint test / train encoding part to pickle
            print('get inference results')
            self.train_lat_rep, self.test_lat_rep, encoded_list, test_encoded_list, keys_list  = self.run_inference_autoencoder()
            infer_path = os.path.join(self.save_path + 'inference/')
            np.save(infer_path +'train_lat_rep.npy', self.train_lat_rep)
            np.save(infer_path +'test_lat_rep.npy', self.test_lat_rep)
            file = open(infer_path + 'encoded_list', 'wb')
            # dump information to that file
            # number of batches, num_dataset, batchsize, h, w, dim
            print('dumping encoded_list to pickle')
            pickle.dump(encoded_list, file)
            file.close()

            test_file = open(infer_path + 'test_encoded_list', 'wb')
            # dump information to that file
            # number of batches, num_dataset, batchsize, h, w, dim
            print('dumping test_encoded_list to pickle')
            pickle.dump(test_encoded_list, test_file)
            test_file.close()




        # calculate performance using only cells that intersect with city boundary
        # do evaluation using matrix format
        # self.evaluation()

        # convert predicted_vals to pandas dataframe with timestamps
        # self.conv3d_predicted = self.arr_to_df()


    def run_autoencoder(self):
        tf.reset_default_graph()
        # self, channel, time_steps, height, width
        predictor = Autoencoder(self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                        self.intersect_pos_set,
                     self.demo_mask_arr, self.dim, self.grouping_dict,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        # (9337, 1, 32, 20, 1)
        train_lat_rep, test_lat_rep, encoded_list, keys_list = predictor.train_autoencoder(
                        self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict, self.train_hours,
                         self.demo_mask_arr, self.save_path, self.dim, self.grouping_dict,
                use_pretrained =  self.use_pretrained, pretrained_ckpt_path = self.pretrained_ckpt_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return train_lat_rep, test_lat_rep, encoded_list, keys_list




    # run training and testing together
    def run_resume_training(self):
        tf.reset_default_graph()
        # self, channel, time_steps, height, width
        predictor = Autoencoder(self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                        self.intersect_pos_set,
                     self.demo_mask_arr, self.dim, self.grouping_dict,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        train_lat_rep, test_lat_rep, encoded_list, keys_list = predictor.train_autoencoder(
                        self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict, self.train_hours,
                         self.demo_mask_arr, self.save_path, self.dim, self.grouping_dict,
                         True, self.checkpoint_path,
                          self.use_pretrained, self.pretrained_ckpt_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)


        # train_lat_rep, test_lat_rep = predictor.train_autoencoder_from_checkpoint(
        #                 self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict, self.train_hours,
        #                  self.demo_mask_arr, self.save_path, self.dim, self.checkpoint_path, self.grouping_dict,
        #              epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return train_lat_rep, test_lat_rep, encoded_list, keys_list



    # run inference only
    def run_inference_autoencoder(self):
        tf.reset_default_graph()
        predictor = Autoencoder(self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                        self.intersect_pos_set,
                     self.demo_mask_arr, self.dim, self.grouping_dict,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        train_lat_rep, test_lat_rep, encoded_list, test_encoded_list, keys_list = predictor.inference_autoencoder(
                        self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict, self.train_hours,
                         self.demo_mask_arr, self.save_path, self.dim, self.grouping_dict,
                        self.checkpoint_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return train_lat_rep, test_lat_rep, test_encoded_list, encoded_list, keys_list




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
