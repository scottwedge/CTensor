# updated April 27, 2020
# added permutation for training / test data
# updated April 29, 2020
# added gradnorm similar to
# http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf
# https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_segnet_mtan.py

# May 6, 2020
# -- use outer product
# -- use adaptive weights for datasets


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
import random
import pickle

import tensorflow.python.keras
import tensorflow.contrib.keras as keras
from tensorflow.python.keras import backend as K

import copy
from random import shuffle



HEIGHT = 32
WIDTH = 20
TIMESTEPS = 24

BATCH_SIZE = 32
# actually epochs
TRAINING_STEPS = 50
# TRAINING_STEPS = 1
LEARNING_RATE = 0.003
# HOURLY_TIMESTEPS = 168
HOURLY_TIMESTEPS = 24
DAILY_TIMESTEPS = 1
THREE_HOUR_TIMESTEP = 56

ALPH = 3
STARTER_ITERATION = 50
T = 5


def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)


def generate_fixlen_timeseries(rawdata_arr, timestep = TIMESTEPS):
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
# def create_mini_batch_1d(start_idx, end_idx,  data_1d):
#     # data_3d : (45984, 32, 20, ?)
#     # data_1d: (45984, ?)
#     # data_2d: (32, 20, ?)
#     test_size = end_idx - start_idx
#
#     # test_data_1d = data_1d[start_idx:end_idx + 168 - 1,:]
#     test_data_1d = data_1d[start_idx:end_idx + TIMESTEPS - 1,:]
#     test_data_1d_seq = generate_fixlen_timeseries(test_data_1d)
#     test_data_1d_seq = np.swapaxes(test_data_1d_seq,0,1)
#     # (168, batchsize, dim)
#     return test_data_1d_seq

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
# def create_mini_batch_3d(start_idx, end_idx,data_3d, timestep):
#     # data_3d : (45984, 32, 20, ?)
#     # data_1d: (45984, ?)
#     # data_2d: (32, 20, ?)
#
#     test_size = end_idx - start_idx
#     # handle different time frame
#     test_data_3d = data_3d[start_idx :end_idx + timestep - 1, :, :]
#     test_data_3d_seq = generate_fixlen_timeseries(test_data_3d, timestep)
#     test_data_3d_seq = np.expand_dims(test_data_3d_seq, axis=4)
#     test_data_3d_seq = np.swapaxes(test_data_3d_seq,0,1)
#     # (timestep (168/56/7), batchsize, 32, 20, 1)
#     return test_data_3d_seq


def create_mini_batch_3d(start_index_list, start_idx, end_idx, data_3d, timestep):
    raw_seq_list = list()
    # arr_shape: [# of timestamps, w, h]
    arr_shape = data_3d.shape
    for start in start_index_list[start_idx: end_idx]:
        end = start + timestep
        # print('3d:start: end',  start, end)
        temp_seq = data_3d[start: end]
        raw_seq_list.append(temp_seq)
    raw_seq_arr = np.array(raw_seq_list)
    raw_seq_arr = np.expand_dims(raw_seq_arr, axis=4)
    return raw_seq_arr


def create_mini_batch_1d(start_index_list, start_idx, end_idx, data_1d):
    raw_seq_list = list()
    # arr_shape: [# of timestamps, w, h]
    arr_shape = data_1d.shape
    for start in start_index_list[start_idx: end_idx]:
        end = start + TIMESTEPS
        # print('1d: start: end',  start, end)
        temp_seq = data_1d[start: end]
        raw_seq_list.append(temp_seq)
    raw_seq_arr = np.array(raw_seq_list)

    return raw_seq_arr



def generate_fixlen_timeseries_nonoverlapping(rawdata_arr, timestep = TIMESTEPS):
    raw_seq_list = list()
    # arr_shape: [# of timestamps, w, h]
    arr_shape = rawdata_arr.shape
    # e.g., 50 (batchsize * timestep), or 21 (leftover)
    for i in range(0, arr_shape[0], timestep):
        start = i
        end = i+ (timestep )
        # ignore if a small sequence of data that is shorter than timestep
        if end <= arr_shape[0]:
            # temp_seq = rawdata_arr[start: end, :, :]
            temp_seq = rawdata_arr[start: end]
            raw_seq_list.append(temp_seq)

    raw_seq_arr = np.array(raw_seq_list)
    raw_seq_arr = np.swapaxes(raw_seq_arr,0,1)
    return raw_seq_arr


# create non-overlapping sequences
# create a batchsize (e.g., 32) of 24-hour non-overlapping sequences
# end_idx - start_idx = batchsize * TIMESTEPS
def create_mini_batch_1d_nonoverlapping(start_idx, end_idx,  data_1d):
    # data_3d : (45984, 32, 20, ?)
    # data_1d: (45984, ?)
    # data_2d: (32, 20, ?)
    test_size = end_idx - start_idx
    # test_data_1d = data_1d[start_idx:end_idx + 168 - 1,:]
    test_data_1d = data_1d[start_idx:end_idx,:]
    test_data_1d_seq = generate_fixlen_timeseries_nonoverlapping(test_data_1d)
    test_data_1d_seq = np.swapaxes(test_data_1d_seq,0,1)
    # (168, batchsize, dim)
    return test_data_1d_seq


# output: batchsize, h, w, dim
def create_mini_batch_2d_nonoverlapping(start_idx, end_idx,  data_2d):
    # data_3d : (45984, 32, 20, ?)
    # data_1d: (45984, ?)
    # data_2d: (32, 20, ?)
    test_size = end_idx - start_idx
    test_data_2d = np.expand_dims(data_2d, axis=0)
    if int(test_size / TIMESTEPS) == BATCH_SIZE:

        test_data_2d = np.tile(test_data_2d,(BATCH_SIZE,1,1,1))
    else:
        test_data_2d = np.tile(test_data_2d,(int(test_size / TIMESTEPS),1,1,1))
    # (batchsize, 32, 20, 20)
    return test_data_2d


def create_mini_batch_3d_nonoverlapping(start_idx, end_idx, data_3d, timestep):
    # data_3d : (45984, 32, 20, ?) or (1, 45840, 32, 20)
    # data_1d: (45984, ?)
    # data_2d: (32, 20, ?)
    test_size = end_idx - start_idx
    # else:  # 911 data
    test_data_3d = data_3d[start_idx :end_idx, :, :]
    test_data_3d_seq = generate_fixlen_timeseries_nonoverlapping(test_data_3d, timestep)
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
    prefix_list = ['1d_data_process_', '2d_data_process_', '3d_data_process_',
             ]
    # '1d_data_reconstruct_', '2d_data_reconstruct_', '3d_data_reconstruct_'
    suffix_list = keys_1d + keys_2d + keys_3d
    for prefix in prefix_list:
        for suffix in suffix_list:
            scopes_to_reserve.append(prefix + suffix)
    return scopes_to_reserve


# get names of variable_scopes to be restored
# from a checkpoint path dict:  {key: path}
# e.g., precipitation: path to checkpoint
def get_scopes_to_restore_for_eachdataset(key, keys_1d, keys_2d, keys_3d):
    scopes_to_reserve = []
    # prefix_list = ['1d_data_process_', '2d_data_process_', '3d_data_process_',
    #          ]
    # for key in suffix_list:
    if key in keys_1d:
        scopes_to_reserve.append('1d_data_process_' + key)
    if key in keys_2d:
        scopes_to_reserve.append('2d_data_process_' + key)
    if key in keys_3d:
        scopes_to_reserve.append('3d_data_process_' + key)
    return scopes_to_reserve


# for gradnorm, get all parameters from AE,
# except for loss weights.
def get_parameters_from_model():
    scopes_to_discard = 'lossweight'
    variables_to_restore = []
    for v in tf.trainable_variables():
        # # DEBUG:
        if v.name.split(':')[0].split('/')[0].startswith(scopes_to_discard):
            print("Variables discard: %s" % v.name)
        if not v.name.split(':')[0].split('/')[0].startswith(scopes_to_discard):
            print("Variables in models: %s" % v.name)
            variables_to_restore.append(v)
    return variables_to_restore



def get_parameters_from_sharedlayers(model_parameters):
    scopes_to_reserve = 'last_shared_layer'
    variables_to_restore = []
    for v in model_parameters:
        if v.name.split(':')[0].split('/')[1] == scopes_to_reserve:
            print("last layer parameters: %s" % v.name)
            variables_to_restore.append(v)
    return variables_to_restore


class Autoencoder:
    # input_dim = 1, seq_size = 168,
    def __init__(self, rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
               rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
                   intersect_pos_set,
                    demo_mask_arr, dim,
                    channel, time_steps, height, width):

        self.time_steps = time_steps
        self.width = width
        self.height = height
        self.channel = channel  # 27
        self.dim  = dim # default = 1, it is the dimension of latent representation
        # this is usefor Batch normalization.
        # https://towardsdatascience.com/pitfalls-of-batch-norm-in-tensorflow-and-sanity-checks-for-training-networks-e86c207548c8
        self.is_training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, trainable=False)
        # globale step for Lgrad
        # self.Lgrad_global_step = tf.Variable(0, trainable=False)
        self.dataset_keys = list(rawdata_1d_dict.keys()) + list(rawdata_2d_dict.keys()) + list(rawdata_3d_dict.keys())

        self.rawdata_1d_tf_x_dict = {}
        self.rawdata_1d_tf_y_dict = {}
        self.number_of_tasks = len(rawdata_1d_dict)+ len(rawdata_2d_dict) + len(rawdata_3d_dict)
        if len(rawdata_1d_dict) != 0:
            # rawdata_1d_dict
            for k, v in rawdata_1d_dict.items():
                dim = v.shape[-1]
                # self.rawdata_1d_tf_x_dict[k] = tf.placeholder(tf.float32, shape=[None,TIMESTEPS, dim])
                self.rawdata_1d_tf_y_dict[k] = tf.placeholder(tf.float32, shape=[None,TIMESTEPS, dim])

            for k, v in rawdata_1d_corrupted_dict.items():
                dim = v.shape[-1]
                self.rawdata_1d_tf_x_dict[k] = tf.placeholder(tf.float32, shape=[None,TIMESTEPS, dim])


        # 2d
        self.rawdata_2d_tf_x_dict = {}
        self.rawdata_2d_tf_y_dict = {}
        if len(rawdata_2d_dict) != 0:
            # rawdata_1d_dict
            for k, v in rawdata_2d_dict.items():
                dim = v.shape[-1]
                # self.rawdata_2d_tf_x_dict[k] = tf.placeholder(tf.float32, shape=[None, height, width, dim])
                self.rawdata_2d_tf_y_dict[k] = tf.placeholder(tf.float32, shape=[None, height, width, dim])

            for k, v in rawdata_2d_corrupted_dict.items():
                dim = v.shape[-1]
                self.rawdata_2d_tf_x_dict[k] = tf.placeholder(tf.float32, shape=[None, height, width, dim])


        # -------- 3d --------------#
        self.rawdata_3d_tf_x_dict = {}
        self.rawdata_3d_tf_y_dict = {}
        if len(rawdata_3d_dict) != 0:
            for k, v in rawdata_3d_dict.items():
                # self.rawdata_3d_tf_x_dict[k] = tf.placeholder(tf.float32, shape=[None,HOURLY_TIMESTEPS, height, width, 1])
                self.rawdata_3d_tf_y_dict[k] = tf.placeholder(tf.float32, shape=[None,HOURLY_TIMESTEPS, height, width, 1])

            for k, v in rawdata_3d_corrupted_dict.items():
                self.rawdata_3d_tf_x_dict[k] = tf.placeholder(tf.float32, shape=[None,HOURLY_TIMESTEPS, height, width, 1])

        ############# variables for gradnorm ######################

        # weights for loss of each dataset
        self.weights_dict = {}
        if len(rawdata_1d_dict) != 0:
            for k, v in rawdata_1d_dict.items():
                var_name =  k
                # self.weights_dict[var_name] = tf.Variable(1, trainable = True, dtype = tf.float32, name = var_name)
                self.weights_dict[var_name] = tf.placeholder(tf.float32, shape=[], name = var_name)
        if len(rawdata_2d_dict) != 0:
            for k, v in rawdata_2d_dict.items():
                var_name =  k
                self.weights_dict[var_name] = tf.placeholder(tf.float32, shape=[], name = var_name)
        if len(rawdata_3d_dict) != 0:
            for k, v in rawdata_3d_dict.items():
                var_name =  k
                self.weights_dict[var_name] = tf.placeholder(tf.float32, shape=[], name = var_name)


        # batchloss for each dataset in each step, for computing inverse training rate
        # self.batchloss_dict = {}
        # self.L0_dict = {}
        # if len(rawdata_1d_dict) != 0:
        #     for k, v in rawdata_1d_dict.items():
        #         #self.batchloss_dict[k] =  tf.placeholder(tf.float32, shape=[])
        #         self.L0_dict[k] = tf.placeholder(tf.float32, shape=[])
        #
        # if len(rawdata_2d_dict) != 0:
        #     for k, v in rawdata_2d_dict.items():
        #         #self.batchloss_dict[k] =  tf.placeholder(tf.float32, shape=[])
        #         self.L0_dict[k] = tf.placeholder(tf.float32, shape=[])
        # if len(rawdata_3d_dict) != 0:
        #     for k, v in rawdata_3d_dict.items():
        #         #self.batchloss_dict[k] =  tf.placeholder(tf.float32, shape=[])
        #         self.L0_dict[k] = tf.placeholder(tf.float32, shape=[])


    # update on Jan, 2020: change variable_scopes
    # updated on Feb 4, 2020. ensure the output is [None, 168, 32, 20, output_dim]
    def cnn_model(self, x_train_data, is_training, suffix = '', output_dim = 1, keep_rate=0.7, seed=None):
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
            conv3 = tf.layers.conv3d(inputs=conv2, filters= output_dim, kernel_size=[3,3,3], padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

            #
            out = conv3
        # original:output size should be [None, height, width, channel]
        # output size should be [None, 168, height, width, channel]
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
            # conv2: change from 16 to 32
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
        # output size should be [None, height, width, 1]
        return out


    '''
    input: 1d feature tensor: height * width * # of features
                (batchsize, # of timestamp, channel), e.g., (32, 168,  3)
    output:    [None, 168, output_dim]
              original size: (batchsize, 1), deprecated in this version
    '''
    # update: keep the output shape [None, 168, output_dim]
    # (batchsize, 168, # of features)
    def cnn_1d_model(self, x_1d_train_data, is_training, suffix = '', output_dim =1, seed=None):
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
            # conv2 change from 16 to 32
            conv2 = tf.layers.conv1d(conv1, 32, 3,padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            # Average Pooling   None, 168,16  -> None, 1, 16
            # conv2 = tf.layers.average_pooling1d( conv2, 168, 1, padding='valid')

        # with tf.name_scope("1d_layer_b"):
            conv3 = tf.layers.conv1d(
                      inputs=conv2,
                      filters=output_dim, # switch from 1 to 3
                      kernel_size=1,
                      padding="same",
                      activation=my_leaky_relu
                      #reuse = tf.AUTO_REUSE
                )

            # (batchsize, 168, dim)
            out = conv3
        return out



    # [batchsize, height, width, dim] -> recontruct to [None, DAILY_TIMESTEPS, height, width, 1]
    # update: [None, 168, 32, 20, dim_decode] -> recontruct to [None, DAILY_TIMESTEPS, height, width, 1]
    def reconstruct_3d(self, latent_fea, timestep, is_training, suffix = ''):
        var_scope = "3d_data_reconstruct_" + suffix
        with tf.variable_scope(var_scope):
            padding = 'SAME'
            stride = [1,1,1]

                # [None, 168, 32, 20, dim_decode] ->  [None, DAILY_TIMESTEPS, height, width, 1]
            conv1 = tf.layers.conv3d(inputs=latent_fea, filters=16, kernel_size=[3,3,3], padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)
                # conv => 16*16*16
            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)
                # pool => 8*8*8

                # [None, 168, 32, 20, total_dim]->  [None, 168, 32, 20, dim]
            conv3 = tf.layers.conv3d(inputs=conv2, filters= 1, kernel_size=[3,3,3], padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

            output = conv3

        return output


    # previous: [None, 32, 20, dim ] -> recontruct to data_2d: (None, 32, 20, dim_2d)
    # update: [None, 168, 32, 20, dim_decode] ->  (None, 32, 20, dim_2d)
    def reconstruct_2d(self, latent_fea, dim_2d, is_training, suffix = ''):
        var_scope = "2d_data_reconstruct_" + suffix
        with tf.variable_scope(var_scope):
            padding = 'SAME'

            #Average Pooling  [None, 168, 32, 20, dim_decode] ->  (None, 1, 32, 20, dim_decode)
            # https://www.tensorflow.org/api_docs/python/tf/compat/v1/layers/average_pooling3d
            conv1 = tf.layers.average_pooling3d(latent_fea, [TIMESTEPS, 1, 1], [1,1,1], padding='valid')
            # (None, 1, 32, 20, dim_decode)  -> (None,  32, 20, dim_decode)
            conv1 = tf.squeeze(conv1, axis = 1)

            conv2 = tf.layers.conv2d(conv1, 16, 3, padding='same',activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            conv3 = tf.layers.conv2d(conv2, 32, 3, padding='same',activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)
            # [None, 32, 20, 16]  -> [None,32, 20 32]

            conv4 = tf.layers.conv2d(conv3, dim_2d, 3, padding='same',activation=None)
            conv4 = tf.layers.batch_normalization(conv4, training=is_training)
            conv4 = tf.nn.leaky_relu(conv4, alpha=0.2)

            #[None,32, 20 32] -> [None, 32, 20, dim_2d]
        return conv4


    # previsou: [None, 32, 20, dim ] -> recontruct to [None,168, dim_1d]
    # new: [None, 168, 32, 20, dim_decode] ->  [None,168, dim_1d]
    def reconstruct_1d(self, latent_fea, dim_1d, is_training, suffix= ''):
        var_scope = "1d_data_reconstruct_" + suffix
        with tf.variable_scope(var_scope):
            # [None, 168, 32, 20, dim_decode] ->  [None,168, 1, 1, dim_decode]
            conv1 = tf.layers.average_pooling3d(latent_fea, [1, HEIGHT, WIDTH], [1,1,1], padding='valid')
            # [None,168, 1, 1, dim_decode]  -> [None,168, 1, dim_decode]
            conv1 = tf.squeeze(conv1, axis = 2)
            # [None,168, 1,  dim_decode]  -> [None,168, dim_decode]
            conv1 = tf.squeeze(conv1, axis = 2)

            conv2 = tf.layers.conv1d(conv1, 16, 3, padding='same',activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

            #  Convolution Layer with 64 filters and a kernel size of 3
            # output shape: None, 168,16
            # conv2 change from 16 to 32
            conv3 = tf.layers.conv1d(conv2, 32, 3,padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

            conv4 = tf.layers.conv1d(conv3, dim_1d, 3,padding='same', activation=None)
            conv4 = tf.layers.batch_normalization(conv4, training=is_training)
            conv4 = tf.nn.leaky_relu(conv4, alpha=0.2)

        return conv4


    # after 1d cnn encoding layer, concat the 1d latent rep together
    # feed a list of 1d latent rep into cnn_1d_fuse, compress and output
    # (batchsize, 168, # of features)
    def cnn_1d_fuse(self, feature_map_list, is_training, suffix = '', output_dim =1, seed=None):
        # var_scope = "1d_data_process_" + suffix
        output_dim = int(len(feature_map_list) / 3) + 1
        with tf.variable_scope('cnn_1d_fuse'):
            fuse_feature =tf.concat(axis=-1,values=feature_map_list)
            print('fuse_feature.shape: ', fuse_feature.shape)
                        # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv1d(fuse_feature, 32, 3, padding='same',activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

        # with tf.name_scope("1d_layer_b"):
            conv2 = tf.layers.conv1d(
                      inputs=conv1,
                      filters=output_dim, # switch from 1 to 3
                      kernel_size=1,
                      padding="same",
                      activation=my_leaky_relu
                      #reuse = tf.AUTO_REUSE
                )
            # conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            # (batchsize, 168, dim)
            out = conv2

            prediction_1d = tf.expand_dims(out, 2)
            prediction_1d = tf.expand_dims(prediction_1d, 2)
            prediction_1d_expand = tf.tile(prediction_1d, [1, 1, HEIGHT,
                                                    WIDTH ,1])
        return prediction_1d_expand



    '''
    input: 2d feature tensor: height * width * # of features (batchsize, 32, 20, 4)
    output: (32, 20, 1)
    '''
    def cnn_2d_fuse(self, feature_map_list, is_training, suffix = '', output_dim = 1, seed=None):
        # var_scope = "2d_data_process_" + suffix
        output_dim = int(len(feature_map_list) / 3) + 1
        with tf.variable_scope('cnn_2d_fuse'):
            fuse_feature =tf.concat(axis=-1,values=feature_map_list)
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(fuse_feature, 32, 3, padding='same',activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

        # output should be (?, 32, 20, 1)
        # with tf.name_scope("2d_layer_b"):
            conv2 = tf.layers.conv2d(
                      inputs=conv1,
                      filters=output_dim,
                      kernel_size=[1, 1],
                      padding="same",
                      activation=my_leaky_relu
                      #reuse = tf.AUTO_REUSE
                )
            # conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            out = conv2
            prediction_2d = tf.expand_dims(out, 1)
            prediction_2d_expand = tf.tile(prediction_2d, [1, TIMESTEPS, 1,
                                                    1 ,1])
        # output size should be [None, height, width, 1]
        return prediction_2d_expand



    def cnn_3d_fuse(self, feature_map_list, is_training, suffix = '', output_dim = 1, seed=None):
        # var_scope = "2d_data_process_" + suffix
        output_dim = int(len(feature_map_list) / 3) + 1
        with tf.variable_scope('cnn_3d_fuse'):
            fuse_feature =tf.concat(axis=-1,values=feature_map_list)
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv3d(inputs=fuse_feature, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

        # output should be (?, 32, 20, 1)
        # with tf.name_scope("2d_layer_b"):
            conv2 = tf.layers.conv3d(
                      inputs=conv1,
                      filters=output_dim,
                      kernel_size=[1, 1, 1],
                      padding="same",
                      activation=my_leaky_relu
                )
            # conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            out = conv2
        # output size should be [None, height, width, 1]
        return out







    # element product fusion updated on May 5
    def fuse_and_train(self, first_level_output_1d, first_level_output_2d, first_level_output_3d, is_training, suffix = '', dim=3):
        var_scope = 'fusion_layer_'+ suffix
        with tf.variable_scope(var_scope):
            fused_1d = self.cnn_1d_fuse(first_level_output_1d, is_training)
            fused_2d = self.cnn_2d_fuse(first_level_output_2d, is_training)
            dim_1d = fused_1d.shape[-1]
            dim_2d = fused_2d.shape[-1]

            fused_1d_extend = tf.tile(fused_1d, [1, 1, 1,1 ,dim_2d])
            fused_2d_extend = tf.tile(fused_2d, [1, 1, 1,1 ,dim_1d])

            fused_1d2d = tf.multiply(fused_1d_extend, fused_2d_extend)
            print('fused_1d2d.shape: ', fused_1d2d.shape)

            if len(first_level_output_3d)!=0:

                #fused_3d =tf.concat(axis=-1,values=first_level_output_3d)
                fused_3d = self.cnn_3d_fuse(first_level_output_3d, is_training)
                dim_3d = fused_3d.shape[-1] # temporarily only feed one 3d datasets
                fused_3d_extend = tf.tile(fused_3d, [1, 1, 1,1 ,dim_2d * dim_1d])
                fused_1d2d_extend = tf.tile(fused_1d2d, [1, 1, 1,1 ,dim_3d])

                fuse_feature = tf.multiply(fused_3d_extend, fused_1d2d_extend)
                print('fuse_feature.shape: ', fuse_feature.shape)
            else:
                fuse_feature = fused_1d2d

            # fuse_feature =tf.concat(axis=-1,values=feature_map_list)
            # print('fuse_feature.shape: ', fuse_feature.shape)

            # [None, 168, 32, 20, total_dim]->  [None, 168, 32, 20, 16]
            conv1 = tf.layers.conv3d(inputs=fuse_feature, filters=16, kernel_size=[3,3,3], padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)
            # conv => 16*16*16
            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)
            # pool => 8*8*8

            # [None, 168, 32, 20, total_dim]->  [None, 168, 32, 20, dim]
            conv3 = tf.layers.conv3d(inputs=conv2, filters= dim, kernel_size=[3,3,3], padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

            out = conv3
            print('latent representation shape: ',out.shape)
            # output size should be [batchsize, height, width, dim]
            # updated: output size should be [batchsize, 168, height, width, dim]
        return out




    # take a list of feature maps, combine them through stacking
    # continue to train the stacked feature maps using several conv layers.
    # output a latent feature with specified dim

    # update input shape for
    # 3d: [None, 168, 32, 20, dim]
    # 2d: [height, width, dim]
    # 1d: [None, 168, dim]
    # all expand to shape [None, 168, 32, 20, total_dim]
    # use 3d conv instead of 2d
    def fuse_and_train_concatenation(self, feature_map_list, is_training, suffix = '', dim=3):
        var_scope = 'fusion_layer_'+ suffix
        with tf.variable_scope(var_scope):
            fuse_feature =tf.concat(axis=-1,values=feature_map_list)
            print('fuse_feature.shape: ', fuse_feature.shape)

            # [None, 168, 32, 20, total_dim]->  [None, 168, 32, 20, 16]
            conv1 = tf.layers.conv3d(inputs=fuse_feature, filters=16, kernel_size=[3,3,3], padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)
            # conv => 16*16*16
            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)
            # pool => 8*8*8

            # [None, 168, 32, 20, total_dim]->  [None, 168, 32, 20, dim]
            conv3 = tf.layers.conv3d(inputs=conv2, filters= dim, kernel_size=[3,3,3],
                        padding='same', activation=None, name = 'last_shared_layer')
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

            out = conv3
            print('latent representation shape: ',out.shape)
            # output size should be [batchsize, height, width, dim]
            # updated: output size should be [batchsize, 168, height, width, dim]
        return out


    # take a latent fea, decode into [batchsize, 32, 20, dim_decode]
    # update: latent_fea: [batchsize, 168, height, width, dim]
    #         -> [None, 168, 32, 20, dim_decode]
    def branching(self, latent_fea, dim_decode, is_training, suffix = ''):
        var_scope = 'branching_'+ suffix
        with tf.variable_scope(var_scope):
            padding = 'SAME'
            stride = [1, 1]

            conv1 = tf.layers.conv3d(inputs=latent_fea, filters=16, kernel_size=[3,3,3], padding='same', activation=None)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)
            # conv => 16*16*16
            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3], padding='same', activation=None)
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)
            # pool => 8*8*8

            # [None, 168, 32, 20, total_dim]->  [None, 168, 32, 20, dim]
            conv3 = tf.layers.conv3d(inputs=conv2, filters= dim_decode, kernel_size=[3,3,3], padding='same', activation=None)
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

            # previous [None,32, 20 32] -> [None, 32, 20, dim_2d]
            # [batchsize, 168, height, width, dim] -> [None, 168, 32, 20, dim_decode]
            return conv3



    def train_autoencoder(self, rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
                  rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
                    train_hours,
                     demo_mask_arr, save_folder_path, dim,
                     resume_training = False, checkpoint_path = None,
                      use_pretrained = False, pretrained_ckpt_path_dict = None,
                      alph = ALPH,
                       epochs=1, batch_size=16):
        starter_learning_rate = LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                       5000, 0.96, staircase=True)

        #########################  ENCODE ##############################
        keys_list = []
        first_order_encoder_list = []
        # first level output [dataset name: output]
        # first_level_output = dict()
        first_level_output_1d = dict()
        first_level_output_2d = dict()
        first_level_output_3d = dict()

        for k, v in self.rawdata_1d_tf_x_dict.items():
            # (batchsize, 168, # of features) should expand to (batchsize, 168, 32, 20, # of features)
            prediction_1d = self.cnn_1d_model(v, self.is_training, k)
            first_level_output_1d[k] = prediction_1d
            keys_list.append(k)
            first_order_encoder_list.append(prediction_1d)

        for k, v in self.rawdata_2d_tf_x_dict.items():
            # [None, height, width, 1] -> [None, 168, height, width, 1]
            prediction_2d = self.cnn_2d_model(v, self.is_training, k)
            keys_list.append(k)
            first_level_output_2d[k] = prediction_2d
            first_order_encoder_list.append(prediction_2d)


        for k, v in self.rawdata_3d_tf_x_dict.items():
            prediction_3d = self.cnn_model(v, self.is_training, k)
            first_level_output_3d[k] = prediction_3d
            first_order_encoder_list.append(prediction_3d)
            keys_list.append(k)

        # dim: latent fea dimension
        # latent_fea = self.model_fusion(med_res_3d, med_res_2d, med_res_1d, dim, self.is_training)
        # latent_fea = self.fuse_and_train(list(first_level_output.values()),  self.is_training, '1', dim)
        latent_fea = self.fuse_and_train(list(first_level_output_1d.values()),
                    list(first_level_output_2d.values()), list(first_level_output_3d.values()),
                    self.is_training, '1', dim)

        print('latent_fea.shape: ', latent_fea.shape) # (?, 32, 20, 3)


        ################### RECONSTRUCT AND COMPUTE COST #####################################
        # recontruction
        print('recontruction')
        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(latent_fea)[0],1,1,1])
        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)

        total_loss = 0   #equal weighted
        cost = 0  # weighted
        loss_dict = {} # {dataset name: loss}
        rmse_dict = {}
        reconstruction_dict = dict()  # {dataset name:  reconstruction for this batch}
        weighedloss_dict = {}

        for k, v in self.rawdata_1d_tf_y_dict.items():
            dim_1d = rawdata_1d_dict[k].shape[-1]
            reconstruction_1d = self.reconstruct_1d(latent_fea, dim_1d, self.is_training, k)
            temp_loss = tf.losses.absolute_difference(reconstruction_1d, v)
            total_loss += temp_loss
    #         loss_dict.append(temp_loss)
            loss_dict[k] = temp_loss
            temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_1d, v))
            rmse_dict[k] = temp_rmse
            reconstruction_dict[k] = reconstruction_1d
            # for gradnorm
            weighedloss_dict[k] = temp_loss * self.weights_dict[k]
            cost += weighedloss_dict[k]

        for k, v in self.rawdata_2d_tf_y_dict.items():
            dim_2d = rawdata_2d_dict[k].shape[-1]
            reconstruction_2d = self.reconstruct_2d(latent_fea, dim_2d, self.is_training, k)
            temp_loss = tf.losses.absolute_difference(reconstruction_2d, v, weight)
            total_loss += temp_loss
            loss_dict[k] = temp_loss
            temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_2d, v, weight))
            rmse_dict[k] = temp_rmse
            reconstruction_dict[k] = reconstruction_2d

            weighedloss_dict[k] = temp_loss * self.weights_dict[k]
            cost += weighedloss_dict[k]


        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr_expanded, 1)
            # (?, 1, 32, 20, 1) -> (?, 7, 32, 20, 1)
        for k, v in self.rawdata_3d_tf_y_dict.items():
            timestep_3d = v.shape[1]
            reconstruction_3d = self.reconstruct_3d(latent_fea, timestep_3d, self.is_training, k)
    #         print('reconstruction_3d.shape: ', reconstruction_3d.shape) # (?, 7, 32, 20, 1)
            # 3d weight: (?, 32, 20, 1) -> (?, 7, 32, 20, 1)
            demo_mask_arr_temp = tf.tile(demo_mask_arr_expanded, [1, timestep_3d,1,1,1])
            weight_3d = tf.cast(tf.greater(demo_mask_arr_temp, 0), tf.float32)
            temp_loss = tf.losses.absolute_difference(reconstruction_3d, v, weight_3d)
            total_loss += temp_loss
            loss_dict[k] = temp_loss
            temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_3d, v, weight_3d))
            rmse_dict[k] = temp_rmse
            reconstruction_dict[k] = reconstruction_3d

            weighedloss_dict[k] = temp_loss * self.weights_dict[k]
            cost += weighedloss_dict[k]

        print('total_loss: ', total_loss)

        ##################### OPTIMIZATION ###################################
        with tf.name_scope("training"):
            # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = self.global_step)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            all_model_params = get_parameters_from_model()
            # stardard_grad_lists = optimizer.compute_gradients(cost, var_list=all_model_params)
            stardard_grad_lists = optimizer.compute_gradients(cost)

        # Getting gradients of the last shared layer
        # print('Getting gradients of the last shared layer')
        # gradnorm_dict = {}  # grad_norm for each dataset
        # debug_gradients_dict = {}
        # shared_parameters = get_parameters_from_sharedlayers(all_model_params)
        # for k, v in weighedloss_dict.items():
        #     G1R = tf.gradients(weighedloss_dict[k], shared_parameters)
        #     G1 = tf.norm(G1R[0], name='norm')  # with respect to kernel
        #     gradnorm_dict[k] = G1
        #     debug_gradients_dict[k] = G1R

        # G_avg = tf.div(tf.add_n(list(gradnorm_dict.values())), self.number_of_tasks)

        # Calculating relative losses: batch loss (not weighted) / L0
        # lhat_dict = {}
        # for k, v in loss_dict.items():
        #     lhat_dict[k] = tf.div(v, self.L0_dict[k])
        # lhat_avg = tf.div(tf.add_n(list(lhat_dict.values())), self.number_of_tasks)

        # Calculating relative inverse training rates for tasks
        # inv_rate = {}
        # for k, v in lhat_dict.items():
        #     inv_rate[k] = tf.div(v,lhat_avg)

        # Calculating the constant target for Eq. 2 in the GradNorm paper
        # detach the constant target from computation graph
        # desiredgrad_dict = {}
        # for k, v in inv_rate.items():
        #     desiredgrad_dict[k] = tf.stop_gradient(G_avg* (v**alph))

        # Calculating the gradient loss according to Eq. 2 in the GradNorm paper
        # print('Calculating the gradient loss ')
        # Lgrad_list = []
        # for k, v in gradnorm_dict.items():
        #     Lgrad_list.append(tf.losses.absolute_difference(v, desiredgrad_dict[k]))
        #     # Lgrad_list.append(temp_Lgrad)
        # Lgrad = tf.add_n(Lgrad_list)

        # learning rate for optimizer_Lgrad may not be the same as standard.
        # optimizer_Lgrad = tf.train.AdamOptimizer(learning_rate=0.001)
        # variable weight loss
        # Lgrad_lists = optimizer_Lgrad.compute_gradients(Lgrad, var_list=list(self.weights_dict.values()))

        # Updating loss weights
        # may not use globel step as once applied optimization for Lgrad.
        # the globel step may increase
        # Lgrad_op = optimizer_Lgrad.apply_gradients(Lgrad_lists, global_step= self.Lgrad_global_step, name=None)
        # update model weights
        train_op = optimizer.apply_gradients(stardard_grad_lists, global_step=self.global_step, name=None)


        ###################### PREP for TRAINING ###########################
        train_result = list()
        test_result = list()
        # feature maps of last layer of encoder.
        # used for deriving dataset similarity
        encoded_list = list()
        # random sampling to calculate dataset similarity
        # sample_index = random.sample(range(0, train_hours), num_samples)
        final_reconstruction_dict = {} # temp: only first batch

        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        keys_1d = list(rawdata_1d_dict.keys())
        keys_2d = list(rawdata_2d_dict.keys())
        keys_3d = list(rawdata_3d_dict.keys())
        variables = tf.global_variables()

        # --- dealing with model saver ------ #
        if not use_pretrained:
            saver = tf.train.Saver()
        else:
            print('Restoring saver from pretrained model....')
            saver_dict = {}  # saver name: saver
            for k, cpath in pretrained_ckpt_path_dict.items():
                # train from pretrained model_fusion
                vars_to_restore_dict = {}
                # get scopes_to_reserve
                scopes_to_reserve = get_scopes_to_restore_for_eachdataset(k, keys_1d, keys_2d, keys_3d)
                variable_to_restore = get_variables_to_restore(variables, scopes_to_reserve)
                # print('variable_to_restore in : ', k)
                # print(variable_to_restore)
                # make the dictionary, note that everything here will have “:0”, avoid it.
                for v in variable_to_restore:
                    vars_to_restore_dict[v.name[:-2]] = v
                # only for restoring pretrained model weights
                saver_dict[k] = tf.train.Saver(vars_to_restore_dict)

             # save all variables
            saver = tf.train.Saver()


        ######################## SESSION START ##########################
        config = tf.ConfigProto()
        config.gpu_options.allocator_type ='BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.90
        config.gpu_options.allow_growth=True


        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # ----- if initialized with pretrained weights ----
            if use_pretrained:
                for k, s in saver_dict.items():
                    print('restoring: ', k)
                    s.restore(sess, pretrained_ckpt_path_dict[k])

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

                # load weight_per_epoch
                print('load last saved weight_per_epoch dict')
                with open(save_folder_path + 'weight_per_epoch_dict', 'rb') as handle:
                    weight_per_epoch = pickle.load(handle)
                print('load last L0 dict')
                if os.path.exists(save_folder_path + 'L0_dict'):
                    with open(save_folder_path + 'L0_dict', 'rb') as handle2:
                        L0_dict = pickle.load(handle2)
                else: # use pandas
                    L0_dict = dict()
                    L0_df = pd.read_csv(save_folder_path + 'L0_df.csv', index_col=0)
                    cols = list(L0_df)
                    for c in cols:
                        L0_dict[c] = L0_df[c][0]


            else:
                start_epoch = 0

            # temporary
            # train_hours = 200
            # train_hours: train_start_time = '2014-02-01',train_end_time = '2018-10-31',
            if train_hours%batch_size ==0:
                iterations = int(train_hours/batch_size)
            else:
                iterations = int(train_hours/batch_size) + 1

            ######  for grad norm ###################################
            all_weights = {}  # weight for each dataset
            all_weights = {k: [1] for k in self.dataset_keys}
            # the relative inverse training rate of task i.
            all_inv_rate = {k: [1] for k in self.dataset_keys}
            all_ave_loss_eachdata = {k: [] for k in self.dataset_keys}
            # change weights every epoch, using the first 'starter_interation' iterations
            starter_interation =  STARTER_ITERATION
            # calculated weight this epoch
            if not resume_training:
                print('re-intiate weight for all losses as 1')
                weight_per_epoch = dict(zip(self.dataset_keys, [1]*len(self.dataset_keys)))
                L0_dict = {}  # base cost for each dataset
            # to calculate average inverse training rate
            inv_rate = dict(zip(self.dataset_keys, [1]*len(self.dataset_keys)))
            ############################################################

            for epoch in range(start_epoch, epochs):
                print('Epoch', epoch, 'started', end='')
                start_time = datetime.datetime.now()
                # shuffle data for each epoch, create random index for all data: train + test
                start_index_list = list(range(0, 45984-24))
                shuffle(start_index_list)

                epoch_loss = 0
                epoch_subloss = {}  # ave loss for each dataset
                epoch_subloss = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

                epoch_subrmse = {}  # ave loss for each dataset
                epoch_subrmse = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

                epoch_subweightedloss = {}  # ave loss for each dataset
                epoch_subweightedloss = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))

                # epoch_weights = {}  # weight for each dataset
                # epoch_weights = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))
                final_output = list()
                final_encoded_list = list()

                #########     for changing weights #####################
                # average loss in the first iterations of each epoch for each data
                ave_loss_eachdata = dict(zip(self.dataset_keys, [0]*len(self.dataset_keys)))
                # ave loss / L0
                lhat_dict = {}
                #########################################################

                # mini batch iterations
                for itr in range(iterations):
                    start_idx = itr*batch_size
                    if train_hours < (itr+1)*batch_size:
                        end_idx = train_hours
                    else:
                        end_idx = (itr+1)*batch_size
                    print('Epoch, itr, start_idx, end_idx', epoch, itr, start_idx, end_idx)

                    # create feed_dict
                    feed_dict_all = {}  # tf_var:  tensor
                    # create batches for 1d
                    for k, v in rawdata_1d_dict.items():
                        temp_batch = create_mini_batch_1d(start_index_list, start_idx, end_idx, v)
                        feed_dict_all[self.rawdata_1d_tf_y_dict[k]] = temp_batch

                    for k, v in rawdata_1d_corrupted_dict.items():
                        temp_batch = create_mini_batch_1d(start_index_list, start_idx, end_idx, v)
                        feed_dict_all[self.rawdata_1d_tf_x_dict[k]] = temp_batch

                    # create batches for 2d
                    for k, v in rawdata_2d_dict.items():
                        temp_batch = create_mini_batch_2d(start_idx, end_idx, v)
                        feed_dict_all[self.rawdata_2d_tf_y_dict[k]] = temp_batch

                    for k, v in rawdata_2d_corrupted_dict.items():
                        temp_batch = create_mini_batch_2d(start_idx, end_idx, v)
                        feed_dict_all[self.rawdata_2d_tf_x_dict[k]] = temp_batch

                     # create batches for 3d
                    for k, v in rawdata_3d_dict.items():
                        timestep = TIMESTEPS
                        temp_batch = create_mini_batch_3d(start_index_list, start_idx, end_idx, v, timestep)
                        feed_dict_all[self.rawdata_3d_tf_y_dict[k]] = temp_batch

                    for k, v in rawdata_3d_corrupted_dict.items():
                        timestep = TIMESTEPS
                        temp_batch = create_mini_batch_3d(start_index_list, start_idx, end_idx, v, timestep)
                        feed_dict_all[self.rawdata_3d_tf_x_dict[k]] = temp_batch

                    # is_training: True
                    feed_dict_all[self.is_training] = True
                    for k, v in weight_per_epoch.items():
                        feed_dict_all[self.weights_dict[k]] = v

                    batch_cost, batch_loss_dict, batch_rmse_dict, batch_weighedloss_dict, _= sess.run([total_loss,loss_dict, rmse_dict, weighedloss_dict, train_op],
                                            feed_dict=feed_dict_all)
                    # debug
                    # for k, v in batch_loss_dict.items():
                    #     print('loss: iter: k, v: ',itr, k, v)
                    #     print('weightedloss: iter: k, v: ',itr, k, batch_weighedloss_dict[k])

                    ##################  GRADNORM PART ###############################
                    # base loss at the first iteration. all weights are 1
                    if itr == 0 and epoch == 0:
                        for k,v in batch_weighedloss_dict.items():
                            L0_dict[k] = v

                    # sess.run(stardard_grad_lists, feed_dict= feed_dict_all)
                    # # Getting gradients of the first layers of each tower and calculate their l2-norm
                    # batch_G_avg = sess.run(G_avg, feed_dict= feed_dict_all)

                    # Calculating relative inverse training rates for tasks
                    # print('Calculating relative inverse training rates for tasks')
                    # for k, v in L0_dict.items():
                    #     feed_dict_all[self.L0_dict[k]] = v

                    # get weights for this epoch
                    if itr < starter_interation:
                        for k, v in batch_loss_dict.items():
                            ave_loss_eachdata[k] += v
                    if itr == starter_interation:
                        print('starter_interation: update weights ',  starter_interation)
                        for k, v in ave_loss_eachdata.items():
                            ave_loss_eachdata[k] = float(v / starter_interation)
                            # ave_loss_eachdata of all epochs
                            all_ave_loss_eachdata[k].append(ave_loss_eachdata[k])

                            ##########################################################
                            # compare to ave loss of previous epoch
                            # if epoch == 0:
                            #     lhat_dict[k] = ave_loss_eachdata[k] / L0_dict[k]
                            # else:
                            #     lhat_dict[k] = ave_loss_eachdata[k] / all_ave_loss_eachdata[k][-2]
                            ##########################################################
                            # compare to L0
                            lhat_dict[k] = ave_loss_eachdata[k] / L0_dict[k]

                        #lhat_avg = tf.div(tf.add_n(list(lhat_dict.values())), self.number_of_tasks)
                        lhat_avg = sum(list(lhat_dict.values())) / self.number_of_tasks
                        # inverse training rate for this epoch
                        for k, v in lhat_dict.items():
                            inv_rate[k] = v / lhat_avg
                            all_inv_rate[k].append(inv_rate[k])
                            print('epoch, iter, k, inv_rate :', epoch, itr, k, inv_rate[k])
                        # calculate weights
                        divisor = 0
                        for k, v in inv_rate.items():
                            divisor = divisor + np.exp(v / T)
                        for k, v in inv_rate.items():
                            weight_per_epoch[k] = self.number_of_tasks * (np.exp(v /T) / divisor)
                            all_weights[k].append(weight_per_epoch[k])
                            # self.weights_dict[k] = weight_per_epoch[k]

                    # record inverse learning rate and weights
                    # batch_inv_rate, temp_batch_weights = sess.run([inv_rate, self.weights_dict], feed_dict= feed_dict_all)
                    # for k, v in batch_inv_rate.items():
                    #     print('lhat_list: k,v', k, v)
                    #     all_inv_rate[k].append(v)
                    #     all_weights[k].append(temp_batch_weights['lossweight_' + k])
                    #     print('weights: k,v', k, temp_batch_weights['lossweight_' + k])

                    # optimization
                    # print('update Lgrad')
                    # sess.run(Lgrad_op, feed_dict= feed_dict_all)
                    # print('update training op')
                    # sess.run(train_op, feed_dict= feed_dict_all)

                    # Renormalizing the losses weights
                    # print('Renormalizing the losses weights')
                    # coef = self.number_of_tasks/tf.add_n(list(self.weights_dict.values()))
                    # # temp_batch_weights = self.weights_dict.eval()
                    # for k, v in self.weights_dict.items():
                    #     self.weights_dict[k] = coef*v
                        # ds_name = '_'.join(k.split('_')[1:])
                        # print('ds_name: ', ds_name)

                    ###################  GRADNORM END #####################################
                    # else:  # if itr % 200 ! = 200, dont' use gradnorm
                    #     # sess.run(stardard_grad_lists, feed_dict= feed_dict_all)
                    #     sess.run([train_op], feed_dict= feed_dict_all)
                    #     # DEBUG
                        # batch_gradnorm_dict, batch_debug_gradients_dict,  _ = sess.run([gradnorm_dict, debug_gradients_dict, train_op], feed_dict= feed_dict_all)
                        # for k, v in batch_gradnorm_dict.items():
                        #     print("Iter/Epoch: {}/{}...".format(itr, epoch), 'grad norm:{},{}:'.format(k, v))
                        # print("Iter/Epoch: {}/{}...".format(itr, epoch), 'gradient:{},{}:'.format('temperature', batch_debug_gradients_dict['temperature']))


                    # ############### original normal operations #################
                    # get encoded representation
                    # # [None, 1, 32, 20, 1]
                    batch_output, batch_encoded_list = sess.run([latent_fea, first_order_encoder_list], feed_dict= feed_dict_all)
                    #print('batch_output.shape, : ', batch_output.shape)
                    final_output.extend(batch_output)

                    # temp, only ouput the first batch of reconstruction
                    if itr == 0:
                        batch_reconstruction_dict = sess.run([reconstruction_dict], feed_dict= feed_dict_all)
                        final_reconstruction_dict = copy.deepcopy(batch_reconstruction_dict)

                    # record results every 50 iterations, that is about 900 samples
                    if itr% 50 == 0:
                        final_encoded_list.append(batch_encoded_list)

                    epoch_loss += batch_cost
                    for k, v in epoch_subloss.items():
                        epoch_subloss[k] += batch_loss_dict[k]

                    for k, v in epoch_subrmse.items():
                        epoch_subrmse[k] += batch_rmse_dict[k]

                    for k, v in epoch_subweightedloss.items():
                        epoch_subweightedloss[k] += batch_weighedloss_dict[k]

                    if itr%30 == 0:
                        print("Iter/Epoch: {}/{}...".format(itr, epoch),
                            "Training loss: {:.4f}".format(batch_cost))
                        for k, v in batch_loss_dict.items():
                            print('ave loss, latest loss weight, inv rate for k :', k, v,  weight_per_epoch[k], inv_rate[k])
                            # all_weights[k][-1]


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

                for k, v in epoch_subweightedloss.items():
                    epoch_subweightedloss[k] = v/iterations
                    print('epoch: ', epoch, 'k: ', k, 'mean weighted loss: ', epoch_subweightedloss[k])


                # for k, v in epoch_subgrad.items():
                #     epoch_subgrad[k] = v/iterations
                #     print('epoch: ', epoch, 'k: ', k, 'mean train grad: ', epoch_subgrad[k])
                save_path = saver.save(sess, save_folder_path +'denoising_autoencoder_v2_' +str(epoch)+'.ckpt', global_step=self.global_step)
                # save_path = saver.save(sess, './autoencoder.ckpt')
                print('Model saved to {}'.format(save_path))

                # Testing per epoch
                # -----------------------------------------------------------------
                print('testing per epoch, for epoch: ', epoch)
                # train_hours  = 41616  # train_start_time = '2014-02-01',train_end_time = '2018-10-31'
                test_start = train_hours

                # test_end = rawdata_1d_dict[list(rawdata_1d_dict.keys())[0]].shape[0] -TIMESTEPS  # 45984 - 168
                test_end  = 45960
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
                        temp_batch = create_mini_batch_1d(start_index_list, start_idx, end_idx, v)
                        test_feed_dict_all[self.rawdata_1d_tf_y_dict[k]] = temp_batch

                    for k, v in rawdata_1d_corrupted_dict.items():
                        temp_batch = create_mini_batch_1d(start_index_list, start_idx, end_idx, v)
                        test_feed_dict_all[self.rawdata_1d_tf_x_dict[k]] = temp_batch

                    # create batches for 2d
                    for k, v in rawdata_2d_dict.items():
                        temp_batch = create_mini_batch_2d(start_idx, end_idx, v)
                        test_feed_dict_all[self.rawdata_2d_tf_y_dict[k]] = temp_batch

                    for k, v in rawdata_2d_corrupted_dict.items():
                        temp_batch = create_mini_batch_2d(start_idx, end_idx, v)
                        test_feed_dict_all[self.rawdata_2d_tf_x_dict[k]] = temp_batch

                     # create batches for 3d
                    for k, v in rawdata_3d_dict.items():
                        timestep = TIMESTEPS
                        temp_batch = create_mini_batch_3d(start_index_list, start_idx, end_idx, v, timestep)
                        test_feed_dict_all[self.rawdata_3d_tf_y_dict[k]] = temp_batch


                    for k, v in rawdata_3d_corrupted_dict.items():
                        timestep = TIMESTEPS
                        temp_batch = create_mini_batch_3d(start_index_list, start_idx, end_idx, v, timestep)
                        test_feed_dict_all[self.rawdata_3d_tf_x_dict[k]] = temp_batch

                    # double check this
                    for k, v in weight_per_epoch.items():
                        test_feed_dict_all[self.weights_dict[k]] = v
                    # is_training: True
                    test_feed_dict_all[self.is_training] = True

                    test_batch_cost, test_batch_loss_dict, test_batch_rmse_dict, test_batch_weighedloss_dict = sess.run([total_loss,loss_dict, rmse_dict, weighedloss_dict],
                                                        feed_dict= test_feed_dict_all)

                    # debug
                    for k, v in test_batch_loss_dict.items():
                        print('test loss, weighted loss: ', k, v, test_batch_weighedloss_dict[k])

                    # get encoded representation
                    # # [None, 1, 32, 20, 1]
                    test_batch_output = sess.run([latent_fea], feed_dict= test_feed_dict_all)
                    #print('test_batch_output.shape: ', test_batch_output.shape)
                    test_final_output.extend(test_batch_output)

                    for k, v in test_subloss.items():
                        test_subloss[k] += test_batch_loss_dict[k]

                    for k, v in test_subrmse.items():
                        test_subrmse[k] += test_batch_rmse_dict[k]


                    if itr%10 == 0:
                        print("Iter/Epoch: {}/{}...".format(itr, epoch),
                            "testing loss: {:.4f}".format(test_batch_cost))


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



                train_weighedloss_df = pd.DataFrame([list(epoch_subweightedloss.values())],
                                columns= list(epoch_subweightedloss.keys()))
                train_weighedloss_csv_path = save_folder_path + 'autoencoder_train_epoch_subweightedloss' +'.csv'
                with open(train_weighedloss_csv_path, 'a') as f:
                    train_weighedloss_df.to_csv(f, header=f.tell()==0)


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

                # save weights for loss for each dataset
                weights_df = pd.DataFrame({key: pd.Series(value) for key, value in all_weights.items()})
                weights_csv_path = save_folder_path + 'weights_df' +'.csv'
                with open(weights_csv_path, 'a') as f:
                    weights_df.to_csv(f, header=f.tell()==0)

                # save L0_dict as the base loss for all datasets
                L0_df = pd.DataFrame([list(L0_dict.values())],
                                columns= list(L0_dict.keys()))
                L0_csv_path = save_folder_path + 'L0_df' +'.csv'
                with open(L0_csv_path, 'a') as f:
                    L0_df.to_csv(f, header=f.tell()==0)

                # all_inv_rate: all inverse training rate
                all_inv_rate_df = pd.DataFrame({key: pd.Series(value) for key, value in all_inv_rate.items()})
                all_inv_rate_csv_path = save_folder_path + 'all_inv_rate_df' +'.csv'
                with open(all_inv_rate_csv_path, 'a') as f:
                    all_inv_rate_df.to_csv(f, header=f.tell()==0)

                # save the latest weight_per_epoch
                weight_per_epoch_file = open(save_folder_path + 'weight_per_epoch_dict', 'wb')
                print('dumping weight_per_epoch_file to pickle')
                pickle.dump(weight_per_epoch, weight_per_epoch_file)
                weight_per_epoch_file.close()

                # L0_dict
                L0_dict_file = open(save_folder_path + 'L0_dict', 'wb')
                print('dumping L0_dict_file to pickle')
                pickle.dump(L0_dict, L0_dict_file)
                L0_dict_file.close()


                # save results to txt
                txt_name = save_folder_path + 'denoising_AE_v2_df_' +  '.txt'
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
                    # gradnorm_freq
                    the_file.write('T\n')
                    the_file.write(str(T) + '\n')
                    # the_file.write('alpha\n')
                    # the_file.write(str(alph) + '\n')
                    # the_file.write('keys_list\n')
                    # for item in keys_list:
                    #     the_file.write("%s\n" % item)
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
                    print('final_output.shape: , ', len(final_output))
                    print('final_output[0].shape: ', final_output[0].shape)
                    final_output = np.array(final_output)
                    train_result.extend(final_output)
                    test_final_output = np.array(test_final_output)
                    test_result.extend(test_final_output)
                    encoded_list.extend(final_encoded_list)

            # encoded_res = np.array(test_result)
            train_encoded_res = train_result
            print('train_encoded_res.len: ', len(train_encoded_res))
            train_output_arr = train_encoded_res[0]
            print('train_encoded_res[0].shape: ', train_encoded_res[0].shape)
            # temp mute
            # for i in range(1,len(train_encoded_res)):
            #     train_output_arr = np.concatenate((train_output_arr, train_encoded_res[i]), axis=0)

            test_encoded_res = test_result
            test_output_arr = test_encoded_res[0]
            # temp mute
            # for i in range(1,len(test_encoded_res)):
            #     test_output_arr = np.concatenate((test_output_arr, test_encoded_res[i]), axis=0)

        # This is the latent representation (9337, 1, 32, 20, 1) of training
        return train_output_arr, test_output_arr, encoded_list, keys_list, final_reconstruction_dict



    # do inference using existing checkpoint
    # get latent representation for train and test data altogether
    # the input sequence (24 hours or 168 hours) should have no overlapps
    def get_latent_rep(self, rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
                rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
                    train_hours,
                     demo_mask_arr, save_folder_path, dim,
                     checkpoint_path = None,
                       epochs=1, batch_size=32):

        keys_list = []
        first_order_encoder_list = []
        # first level output [dataset name: output]
        first_level_output_1d = dict()
        first_level_output_2d = dict()
        first_level_output_3d = dict()

        for k, v in self.rawdata_1d_tf_x_dict.items():
            prediction_1d = self.cnn_1d_model(v, self.is_training, k)
            first_level_output_1d[k] = prediction_1d
            keys_list.append(k)
            first_order_encoder_list.append(prediction_1d)

        for k, v in self.rawdata_2d_tf_x_dict.items():
            prediction_2d = self.cnn_2d_model(v, self.is_training, k)
            keys_list.append(k)
            first_level_output_2d[k] = prediction_2d
            first_order_encoder_list.append(prediction_2d)

        for k, v in self.rawdata_3d_tf_x_dict.items():
            prediction_3d = self.cnn_model(v, self.is_training, k)
            # if k == 'seattle911calls':
            first_level_output_3d[k] = prediction_3d
            first_order_encoder_list.append(prediction_3d)
            keys_list.append(k)


        # dim: latent fea dimension
        # latent_fea = self.model_fusion(med_res_3d, med_res_2d, med_res_1d, dim, self.is_training)
        #latent_fea = self.fuse_and_train(list(first_level_output.values()),  self.is_training, '1', dim)
        latent_fea =  self.fuse_and_train(list(first_level_output_1d.values()),
                    list(first_level_output_2d.values()), list(first_level_output_3d.values()),
                    self.is_training, '1', dim)
        print('latent_fea.shape: ', latent_fea.shape) # (?, 32, 20, 3)
        # recontruction
        print('recontruction')
        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr, 0)  # [1, 2]
                # [1, 32, 20, 1]  -> [1, 1, 32, 20, 1]
                # [1, 32, 20, 1] -> [batchsize, 32, 20, 1]
                # batchsize = tf.shape(prediction)[0]
        demo_mask_arr_expanded = tf.tile(demo_mask_arr_expanded, [tf.shape(latent_fea)[0],1,1,1])
        weight = tf.cast(tf.greater(demo_mask_arr_expanded, 0), tf.float32)

        total_loss = 0
        cost = 0
        loss_dict = {} # {dataset name: loss}
        rmse_dict = {}
        reconstruction_dict = dict()  # {dataset name:  reconstruction for this batch}

        for k, v in self.rawdata_1d_tf_y_dict.items():
            dim_1d = rawdata_1d_dict[k].shape[-1]
            reconstruction_1d = self.reconstruct_1d(latent_fea, dim_1d, self.is_training, k)
            temp_loss = tf.losses.absolute_difference(reconstruction_1d, v)
            total_loss += temp_loss
    #         loss_dict.append(temp_loss)
            loss_dict[k] = temp_loss
            temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_1d, v))
            rmse_dict[k] = temp_rmse

            # cost += temp_loss
            reconstruction_dict[k] = reconstruction_1d


        for k, v in self.rawdata_2d_tf_y_dict.items():
            dim_2d = rawdata_2d_dict[k].shape[-1]
            reconstruction_2d = self.reconstruct_2d(latent_fea, dim_2d, self.is_training, k)
            temp_loss = tf.losses.absolute_difference(reconstruction_2d, v, weight)
            total_loss += temp_loss
            loss_dict[k] = temp_loss
            temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_2d, v, weight))
            rmse_dict[k] = temp_rmse

            # cost += temp_loss
            reconstruction_dict[k] = reconstruction_2d


        demo_mask_arr_expanded = tf.expand_dims(demo_mask_arr_expanded, 1)
            # (?, 1, 32, 20, 1) -> (?, 7, 32, 20, 1)


        for k, v in self.rawdata_3d_tf_y_dict.items():
            timestep_3d = v.shape[1]
            reconstruction_3d = self.reconstruct_3d(latent_fea, timestep_3d, self.is_training, k)
            demo_mask_arr_temp = tf.tile(demo_mask_arr_expanded, [1, timestep_3d,1,1,1])
            weight_3d = tf.cast(tf.greater(demo_mask_arr_temp, 0), tf.float32)
            temp_loss = tf.losses.absolute_difference(reconstruction_3d, v, weight_3d)
            total_loss += temp_loss
            loss_dict[k] = temp_loss
            temp_rmse = tf.sqrt(tf.losses.mean_squared_error(reconstruction_3d, v, weight_3d))
            rmse_dict[k] = temp_rmse
            reconstruction_dict[k] = reconstruction_3d



        train_result = list()
        test_result = list()
        encoded_list = list()  # output last layer of encoded for further grouping
        test_encoded_list = list()
        final_reconstruction_dict = {} # temp: only first batch

        save_folder_path = os.path.join(save_folder_path, 'latent_rep/')
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
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

            test_start = train_hours
            test_end  = 45960
            #test_end = rawdata_1d_dict[list(rawdata_1d_dict.keys())[0]].shape[0] -TIMESTEPS  # 45984 - 168
            test_len = test_end - test_start  # 4200
            print('test_start: ', test_start) # 41616
            print('test_end: ', test_end)  # 45960
            print('test_len: ', test_len) #  4200 for 168, and 4344 for 24
            total_len = test_len + train_hours
            print('total_len: ', total_len)  # 45960

            # nonoverlapping sequences
            # e.g., train_hours = 100. batchsize = 10, time = 5
            # iter = 2. if train_hours = 121. iter = 3. ignore the last 1 time_step
            step = batch_size * TIMESTEPS  # 32 * 24 = 768
            if total_len%step ==0:
                iterations = int(total_len/step)
            else:
                iterations = int(total_len/step) + 1  # should be 60 iterations

            print('total iterations: ', iterations)

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
                # e.g. itr = 1, start_idx = 50, end_idx = 100
                start_idx = itr*step
                if total_len < (itr+1)*step:
                    end_idx = total_len
                else:
                    end_idx = (itr+1)*step
                print('itr, start_idx, end_idx', itr, start_idx, end_idx)

                # create feed_dict
                feed_dict_all = {}  # tf_var:  tensor
                    # create batches for 1d
                for k, v in rawdata_1d_dict.items():
                    temp_batch = create_mini_batch_1d_nonoverlapping(start_idx, end_idx, v)
                    # feed_dict_all[self.rawdata_1d_tf_x_dict[k]] = temp_batch
                    feed_dict_all[self.rawdata_1d_tf_y_dict[k]] = temp_batch

                for k, v in rawdata_1d_corrupted_dict.items():
                    temp_batch = create_mini_batch_1d_nonoverlapping(start_idx, end_idx, v)
                    feed_dict_all[self.rawdata_1d_tf_x_dict[k]] = temp_batch


                    # create batches for 2d
                for k, v in rawdata_2d_dict.items():
                    temp_batch = create_mini_batch_2d_nonoverlapping(start_idx, end_idx, v)
                    # feed_dict_all[self.rawdata_2d_tf_x_dict[k]] = temp_batch
                    feed_dict_all[self.rawdata_2d_tf_y_dict[k]] = temp_batch

                for k, v in rawdata_2d_corrupted_dict.items():
                    temp_batch = create_mini_batch_2d_nonoverlapping(start_idx, end_idx, v)
                    feed_dict_all[self.rawdata_2d_tf_x_dict[k]] = temp_batch

                    # create batches for 3d
                for k, v in rawdata_3d_dict.items():
                    # if k == 'seattle911calls':
                    timestep = TIMESTEPS
                    temp_batch = create_mini_batch_3d_nonoverlapping(start_idx, end_idx, v, timestep)
                    feed_dict_all[self.rawdata_3d_tf_y_dict[k]] = temp_batch

                for k, v in rawdata_3d_corrupted_dict.items():
                    timestep = TIMESTEPS
                    temp_batch = create_mini_batch_3d_nonoverlapping(start_idx, end_idx, v, timestep)
                    feed_dict_all[self.rawdata_3d_tf_x_dict[k]] = temp_batch


                feed_dict_all[self.is_training] = True
                batch_cost, batch_loss_dict, batch_rmse_dict = sess.run([total_loss,loss_dict, rmse_dict], feed_dict=feed_dict_all)
                    # get encoded representation
                    # # [None, 1, 32, 20, 1]
                batch_output, batch_encoded_list = sess.run([latent_fea, first_order_encoder_list], feed_dict= feed_dict_all)

                final_output.extend(batch_output)
                final_encoded_list.append(batch_encoded_list)

                # temp, only ouput the first batch of reconstruction
                if itr == 0:
                    batch_reconstruction_dict = sess.run([reconstruction_dict], feed_dict= feed_dict_all)
                    final_reconstruction_dict = copy.deepcopy(batch_reconstruction_dict)


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

            for k, v in epoch_subrmse.items():
                epoch_subrmse[k] = v/iterations

                # save epoch statistics to csv
            ecoch_res_df = pd.DataFrame([[epoch_loss]],
                    columns=[ 'inference_loss'])
            res_csv_path = save_folder_path + 'inference_loss_df' +'.csv'
            with open(res_csv_path, 'a') as f:
                    # Add header if file is being created, otherwise skip it
                ecoch_res_df.to_csv(f, header=f.tell()==0)


            train_sub_res_df = pd.DataFrame([list(epoch_subloss.values())],
                    columns= list(epoch_subloss.keys()))
            train_sub_res_csv_path = save_folder_path + 'inference_loss_sub_res' +'.csv'
            with open(train_sub_res_csv_path, 'a') as f:
                train_sub_res_df.to_csv(f, header=f.tell()==0)


                # --- rmse ------
            train_sub_rmse_df = pd.DataFrame([list(epoch_subrmse.values())],
                    columns= list(epoch_subrmse.keys()))
            train_sub_rmse_csv_path = save_folder_path + 'inference_loss_sub_rmse' +'.csv'
            with open(train_sub_rmse_csv_path, 'a') as f:
                train_sub_rmse_df.to_csv(f, header=f.tell()==0)


            # save results to txt
            txt_name = save_folder_path + 'infer_AE_v2_latent_rep' +  '.txt'
            with open(txt_name, 'w') as the_file:
                    #the_file.write('Only account for grids that intersect with city boundary \n')
                # the_file.write('epoch\n')
                # the_file.write(str(epoch)+'\n')
                the_file.write('dim\n')
                the_file.write(str(self.dim) + '\n')
                the_file.write(' epoch_loss:\n')
                the_file.write(str(epoch_loss) + '\n')
                the_file.write('\n')
                the_file.write('total time of last train epoch\n')
                the_file.write(str(train_time_per_epoch) + '\n')
                the_file.write('time per sample for train\n')
                the_file.write(str(train_time_per_sample) + '\n')
                the_file.write('total time of last test epoch\n')
                the_file.write('keys_list\n')
                for item in keys_list:
                    the_file.write("%s\n" % item)
                the_file.close()

            final_output = np.array(final_output)
            train_result.extend(final_output)

            print('saving output_arr ....')
            train_encoded_res = train_result
            train_output_arr = train_encoded_res[0]
            print('len(train_encoded_res): ', len(train_encoded_res))
            for i in range(1,len(train_encoded_res)):
                print('i, len(train_encoded_res[i]): ',i,  len(train_encoded_res[i]))
                train_output_arr = np.concatenate((train_output_arr, train_encoded_res[i]), axis=0)

            # test_encoded_res = test_result
            # test_output_arr = test_encoded_res[0]
            # for i in range(1,len(test_encoded_res)):
            #     test_output_arr = np.concatenate((test_output_arr, test_encoded_res[i]), axis=0)

        print('train_output_arr.shape: ', train_output_arr.shape)
        # This is the latent representation (9337, 1, 32, 20, 1) of training
        return train_output_arr




'''
fixed lenght time window: 168 hours
'''
class Autoencoder_entry:
    def __init__(self, train_obj,
              rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict,
              rawdata_1d_corrupted_dict, rawdata_2d_corrupted_dict, rawdata_3d_corrupted_dict,
               intersect_pos_set,
                    demo_mask_arr, save_path, dim,
                    HEIGHT, WIDTH, TIMESTEPS, CHANNEL, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                     is_inference = False, checkpoint_path = None,
                     resume_training = False, train_dir = None,
                      use_pretrained = False, pretrained_ckpt_path = None,
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

        self.rawdata_1d_corrupted_dict = rawdata_1d_corrupted_dict
        self.rawdata_2d_corrupted_dict = rawdata_2d_corrupted_dict
        self.rawdata_3d_corrupted_dict = rawdata_3d_corrupted_dict

        self.all_keys = set(list(self.rawdata_1d_dict.keys()) +  list(self.rawdata_2d_dict.keys()) +  list(self.rawdata_3d_dict.keys()))


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


        self.ckpt_path_dict = {}
        if self.use_pretrained:
            # construct checkpoint_dict : key: path
            allfiles = os.listdir(self.pretrained_ckpt_path)
            keys_set = set()
            path_set = set()
            for f in allfiles:
                print(f)
                ds_key = f.split('.')[0]
                ckpt_path = '.'.join(f.split('.')[0:2])
                keys_set.add(ds_key)
                path_set.add(ckpt_path)
                if ds_key in self.all_keys:
                    self.ckpt_path_dict[ds_key] = os.path.join(self.pretrained_ckpt_path, ckpt_path)

            for k, v in self.ckpt_path_dict.items():
                print(k, v)


        # ignore non-intersection cells in test_df
        # this is for evaluation
        # self.test_df_cut = self.test_df.loc[:,self.test_df.columns.isin(list(self.intersect_pos_set))]

        if is_inference == False:
            if resume_training == False:
                    # get prediction results
                    print('training from scratch, and get prediction results')
                    # predicted_vals: (552, 30, 30, 1)
                    self.train_lat_rep, self.test_lat_rep, encoded_list, keys_list, final_reconstruction_dict = self.run_autoencoder()

            else:
                    # resume training
                    print("resume training, and get prediction results")
                    self.train_lat_rep, self.test_lat_rep, encoded_list, keys_list, final_reconstruction_dict = self.run_resume_training()
            np.save(self.save_path +'train_lat_rep.npy', self.train_lat_rep)
            np.save(self.save_path +'test_lat_rep.npy', self.test_lat_rep)
            file = open(self.save_path + 'encoded_list', 'wb')
            # dump information to that file
            # number of batches, num_dataset, batchsize, h, w, dim
            print('dumping encoded_list to pickle')
            pickle.dump(encoded_list, file)
            file.close()

            # dump pickle
            recon_file = open(self.save_path + 'final_reconstruction_dict', 'wb')
            # dump information to that file
            # number of batches, num_dataset, batchsize, h, w, dim
            print('dumping final_reconstruction_dict to pickle')
            pickle.dump(final_reconstruction_dict, recon_file)
            recon_file.close()

        else:
            '''
            # inference only
            print('get inference results')
            self.train_lat_rep, self.test_lat_rep, encoded_list, test_encoded_list, keys_list, final_reconstruction_dict  = self.run_inference_autoencoder()
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

            # dump pickle
            recon_file = open(infer_path + 'final_reconstruction_dict', 'wb')
            # dump information to that file
            # number of batches, num_dataset, batchsize, h, w, dim
            print('dumping final_reconstruction_dict to pickle')
            pickle.dump(final_reconstruction_dict, recon_file)
            recon_file.close()
            '''


            # ----------- get lat rep ---------------------- #
            # run_inference_lat_rep(self):
            print('get inference results')
            self.final_lat_rep  = self.run_inference_lat_rep()
            lat_rep_path = os.path.join(self.save_path + 'latent_rep/')
            np.save(lat_rep_path +'final_lat_rep.npy', self.final_lat_rep)






    def run_autoencoder(self):
        tf.reset_default_graph()
        # self, channel, time_steps, height, width
        predictor = Autoencoder(self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                 self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                 self.intersect_pos_set,
                     self.demo_mask_arr, self.dim,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        # (9337, 1, 32, 20, 1)
        train_lat_rep, test_lat_rep, encoded_list, keys_list, final_reconstruction_dict = predictor.train_autoencoder(
                        self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                        self.train_hours,
                         self.demo_mask_arr, self.save_path, self.dim,
                         use_pretrained =  self.use_pretrained, pretrained_ckpt_path_dict = self.ckpt_path_dict,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return train_lat_rep, test_lat_rep, encoded_list, keys_list, final_reconstruction_dict




    # run training and testing together
    def run_resume_training(self):
        tf.reset_default_graph()
        # self, channel, time_steps, height, width
        predictor = Autoencoder(self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,

                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                         self.intersect_pos_set,
                     self.demo_mask_arr, self.dim,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        train_lat_rep, test_lat_rep, encoded_list, keys_list, final_reconstruction_dict = predictor.train_autoencoder(
                        self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                         self.train_hours,
                         self.demo_mask_arr, self.save_path, self.dim,
                         True, self.checkpoint_path,
                         use_pretrained =  self.use_pretrained, pretrained_ckpt_path_dict = self.ckpt_path_dict,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)


        # train_lat_rep, test_lat_rep = predictor.train_autoencoder_from_checkpoint(
        #                 self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict, self.train_hours,
        #                  self.demo_mask_arr, self.save_path, self.dim, self.checkpoint_path,
        #              epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return train_lat_rep, test_lat_rep, encoded_list, keys_list, final_reconstruction_dict




    # run inference only
    def run_inference_autoencoder(self):
        tf.reset_default_graph()
        predictor = Autoencoder(self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,

                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                         self.intersect_pos_set,
                     self.demo_mask_arr, self.dim,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        train_lat_rep, test_lat_rep, encoded_list, test_encoded_list, keys_list, final_reconstruction_dict = predictor.inference_autoencoder(
                        self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                        self.train_hours,
                         self.demo_mask_arr, self.save_path, self.dim,
                        self.checkpoint_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return train_lat_rep, test_lat_rep, test_encoded_list, encoded_list, keys_list, final_reconstruction_dict


    # run inference to produce a consistent latent rep ready for downstream use
    def run_inference_lat_rep(self):
        tf.reset_default_graph()
        predictor = Autoencoder(self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,

                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                         self.intersect_pos_set,
                     self.demo_mask_arr, self.dim,
                     channel=CHANNEL, time_steps=TIMESTEPS, height=HEIGHT, width = WIDTH)

        train_lat_rep = predictor.get_latent_rep(
                        self.rawdata_1d_dict, self.rawdata_2d_dict, self.rawdata_3d_dict,
                         self.rawdata_1d_corrupted_dict, self.rawdata_2d_corrupted_dict, self.rawdata_3d_corrupted_dict,
                         self.train_hours,
                         self.demo_mask_arr, self.save_path, self.dim,
                        self.checkpoint_path,
                     epochs=TRAINING_STEPS, batch_size=BATCH_SIZE)

        return train_lat_rep



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
