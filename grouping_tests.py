# first-level and second-level grouping
# output a dict of grouping result
# updated Jan 20, 2020

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import json
import sys

import glob # read multiple files
import os
import os.path
from os import getcwd
from os.path import join
from os.path import basename
import collections
import argparse
import pickle

import time
import datetime
from datetime import timedelta
from numpy import transpose
from sklearn.metrics.pairwise import cosine_similarity
import math
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

HEIGHT = 32
WIDTH = 20



# if inside city, assign 1.
# if outside city, assign 0
# turn into numpy array mask, True/ False
def generate_mask_array(intersect_pos_set):
    temp_image = [[1 for i in range(HEIGHT)] for j in range(WIDTH)]
    for i in range(HEIGHT):
        for j in range(WIDTH):
            temp_str = str(j)+'_'+str(i)
            if temp_str in intersect_pos_set:
                temp_image[j][i] = 0

    mask_arr = np.array(temp_image)
    mask_arr = np.rot90(mask_arr)
    # print('mask_arr: ', mask_arr)
    # rawdata_arr = np.moveaxis(rawdata_arr, 0, -1)
    # boolean mask
    return mask_arr



# input a tensor [32, 20, n] or [n, 32, 20]
# remove cells outside city, resulting in, e.g. [500, n]
# return a flatten tensor of 500 * n
# should deal with [24, 32, 20, 3]
def remove_outside_cells(tensor, mask_arr):
    if len(tensor.shape) == 3: # for first level
        demo_mask_arr_expanded = np.expand_dims(mask_arr, 2)  # [1, 2]
                # [1, 32, 20, 1]  -> [1, 1, 32, 20, 1]
                # [1, 32, 20, 1] -> [batchsize, 32, 20, 1]
                # batchsize = tf.shape(prediction)[0]
        demo_mask_arr_expanded = np.tile(demo_mask_arr_expanded, [1,1, tensor.shape[-1]])
        # print('demo_mask_arr_expanded.shape: ', demo_mask_arr_expanded.shape)
        # masked tensor, outside cells should be false / 0
        marr = np.ma.MaskedArray(tensor, mask= demo_mask_arr_expanded)
        compressed_arr = np.ma.compressed(marr)
        return compressed_arr
    if len(tensor.shape) == 4:  # for second level
        demo_mask_arr_expanded = np.expand_dims(mask_arr, 2)  # [1, 2]
        demo_mask_arr_expanded = np.tile(demo_mask_arr_expanded, [1,1, tensor.shape[-1]])
        demo_mask_arr_expanded = np.expand_dims(demo_mask_arr_expanded, 0)  # [1, 2]
        demo_mask_arr_expanded = np.tile(demo_mask_arr_expanded, [tensor.shape[0], 1, 1, 1])
        #print('demo_mask_arr_expanded.shape', demo_mask_arr_expanded.shape)

        marr = np.ma.MaskedArray(tensor, mask= demo_mask_arr_expanded)
        compressed_arr = np.ma.compressed(marr)

        return compressed_arr
    print('Tensor shape error!')
    return None


# - removed outside cells
# 1d feature map shape: [1, 3] -> (24, 1)   originally (24, 1, 1, 1)
# 2d feature map: [32, 20, 1] -> [32, 20, 1]
# 3d feature map: [32, 20, 3]  -> [24, 32, 20, 1]
# compare 2d and 1d: duplicate to 3d and flatten and compare
def first_level_grouping(feature_map_dict, encoded_list_rearrange_concat,
            mask_arr, all_keys, keys_1d, keys_2d, keys_3d =  []):
    height = 32
    width = 20
    relation_all_df = pd.DataFrame(0, columns = all_keys, index = all_keys)
    num_data = len(encoded_list_rearrange_concat[0])
    # num_data
    for n in range(num_data):
        print('n: ', n)
        for ds_name1 in all_keys:
            # 1D case
            if ds_name1 in keys_1d:
                temp_arr1 = feature_map_dict[ds_name1][n,:] # (24, 1, 1, 1)
                # (24, 1) - > [32, 20, 24]
                temp_1d_dup = np.repeat(temp_arr1, 32, axis = 1)
                temp_1d_dup = np.repeat(temp_1d_dup, 20, axis = 2)  # 32, 20, 24, 1

                temp_1d_dup = np.squeeze(temp_1d_dup, axis = -1)  #[24, 32, 20,]
                temp_1d_dup = np.moveaxis(temp_1d_dup, 0, -1) # (32, 20, 24)
                dim1 = temp_arr1.shape[0]  # number of layers in the 2d data
        #         dim1 = temp_arr1.shape[-1]  # number of layers in the 2d data
                for ds_name2 in all_keys:
                    # 1D VS 1D
                    if ds_name2 in keys_1d:
                        ave_SR = 0
        #                 print(ds_name1, ds_name2)
                        temp_arr2 = feature_map_dict[ds_name2][n, :]
                        sim_sparse = cosine_similarity(temp_arr1.reshape(1, -1),
                               temp_arr2.reshape(1, -1))
                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR

                    # 2D VS 1D
                    # 2D:  32, 20, 1
                    # 1D duplicate: 32, 20, 3. This means that there is no spatial variations for 1D
                    # duplicate 2D to 32, 20, 3. This means that there is no temporal variations for 2D
                    # then flatten and compare
                    # This means that there is no temporal variations for 2D
                    if ds_name2 in keys_2d:
                        # temp_arr2 = feature_map_dict[ds_name2][n,:,:,:] # 32, 20, 1
                        # # duplicate to [32, 20, 24]
                        # temp_arr2_mean_dup = np.repeat(temp_arr2, dim1, axis = -1)
                        #
                        # compress_arr2 = remove_outside_cells(temp_arr2_mean_dup, mask_arr) # [32, 20, 24]
                        # compress_arr1 = remove_outside_cells( temp_1d_dup, mask_arr) # [32, 20, 24]
                        #
                        # ave_SR = 0
                        # sim_sparse = cosine_similarity(compress_arr2.reshape(1, -1),
                        #                                            compress_arr1.reshape(1, -1))
                        #
                        # ave_SR = sim_sparse[0][0]
                        # relation_all_df.loc[ds_name1, ds_name2]  += ave_SR
                        relation_all_df.loc[ds_name1, ds_name2]  += 0

                    # 3D VS 1D
                    # duplicate 1D to 3D, flatten and compare
                    if ds_name2 in keys_3d:
                        temp_arr2 = feature_map_dict[ds_name2][n,:,:,:,:] # 3d, e.g. [24, 32, 20, 1]
                        temp_arr2 = np.squeeze(temp_arr2, axis = -1)  #[24, 32, 20]
                        temp_arr2 = np.moveaxis(temp_arr2, 0, -1) # (32, 20, 24)

                        ave_SR = 0 # average spearman correlation

                        compress_arr2 = remove_outside_cells(temp_arr2, mask_arr)
                        compress_arr1 = remove_outside_cells(temp_1d_dup, mask_arr)

                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                                compress_arr2.reshape(1, -1))

                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR


            # 2D case
            if ds_name1 in keys_2d:
                temp_arr1 = feature_map_dict[ds_name1][n,:,:,:]  # [32, 20, 1]
                # print('temp_arr1_mean.shape: ', temp_arr1_mean.shape)
                # temp_arr1_mean_dup = np.repeat(temp_arr1_mean_dup, temp_arr2.shape[-1], axis = 0)

                for ds_name2 in all_keys:
                    # 2D Vs 1D
                    if ds_name2 in keys_1d:
                        relation_all_df.loc[ds_name1, ds_name2]  = relation_all_df.loc[ds_name2, ds_name1]
                    # 2D Vs 2D
                    # take mean along 3rd dimension and compare
                    if ds_name2 in keys_2d:
                        ave_SR = 0 # average spearman correlation
                        temp_arr2 = feature_map_dict[ds_name2][n,:,:,:]

                        compress_arr2 = remove_outside_cells(temp_arr2, mask_arr)
                        compress_arr1 = remove_outside_cells( temp_arr1, mask_arr)

                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                    compress_arr2.reshape(1, -1))
    #                             pearson_coef, p_value = stats.pearsonr(temp_arr1[ :, :, i].ravel(), temp_arr2[ :, :, j].ravel())

                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2] += ave_SR

                    # 2D VS 3D
                    # for 2D feature maps, output 3rd dimension of feature map is 1.
                    # for 3D feature maps, output 3rd dimension is 3
                    # average 3D feature map by 3rd dimension
                    # flatten and compare
                    if ds_name2 in keys_3d:
                        temp_arr2 = feature_map_dict[ds_name2][n,:,:,:,:]     #[24, 32, 20, 1]
                        temp_arr2 = np.squeeze(temp_arr2, axis = -1)  #[24, 32, 20]
                        temp_arr2 = np.moveaxis(temp_arr2, 0, -1) # (32, 20, 24)

                        # average along third dimension
                        temp_arr2_mean = np.mean(temp_arr2, axis = -1)
                        temp_arr2_mean_dup = np.expand_dims(temp_arr2_mean, axis = -1) #[32, 20, 1]


                        compress_arr2 = remove_outside_cells( temp_arr2_mean_dup, mask_arr)
                        compress_arr1 = remove_outside_cells( temp_arr1, mask_arr)

                        ave_SR = 0 # average spearman correlation
                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                   compress_arr2.reshape(1, -1))
                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR

            # 3D
            if ds_name1 in keys_3d:
                temp_arr1 = feature_map_dict[ds_name1][n,:,:,:,:]  # [24, 32, 20, 1]
                temp_arr1 = np.squeeze(temp_arr1, axis = -1)  #[24, 32, 20]
                temp_arr1 = np.moveaxis(temp_arr1, 0, -1) # (32, 20, 24)


                for ds_name2 in all_keys:
                    # 1D
                    if ds_name2 in keys_1d:
                        relation_all_df.loc[ds_name1, ds_name2]  = relation_all_df.loc[ds_name2, ds_name1]
                    # 3D VS 2D
                    if ds_name2 in keys_2d:
                        temp_arr2 = feature_map_dict[ds_name2]

                        relation_all_df.loc[ds_name1, ds_name2]  = relation_all_df.loc[ds_name2, ds_name1]

                    # 3D VS 3D
                    # flatten and compare. Because 3rd dimension contains
                    # temporal information
                    if ds_name2 in keys_3d:
                        temp_arr2 = feature_map_dict[ds_name2][n,:,:,:,:]
                        temp_arr2 = np.squeeze(temp_arr2, axis = -1)  #[24, 32, 20]
                        temp_arr2 = np.moveaxis(temp_arr2, 0, -1) # (32, 20, 24)

                        ave_SR = 0 # average spearman correlation
                        compress_arr2 = remove_outside_cells( temp_arr2, mask_arr)
                        compress_arr1 = remove_outside_cells( temp_arr1, mask_arr)

                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                                                       compress_arr2.reshape(1, -1))

                        ave_SR = float(sim_sparse[0][0])
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR
    relation_all_df = relation_all_df / num_data
    return relation_all_df




# 1d feature map shape: [1, 3] -> (24, 1)   originally (24, 1, 1, 1)
# 2d feature map: [32, 20, 1] -> [32, 20, 1]
# 3d feature map: [32, 20, 3]  -> [24, 32, 20, 1]
# all duplicate to 3d and comapre
def first_level_grouping_simplified(feature_map_dict, encoded_list_rearrange_concat,
            mask_arr, all_keys, keys_1d, keys_2d, keys_3d =  []):
    height = 32
    width = 20
    relation_all_df = pd.DataFrame(0, columns = all_keys, index = all_keys)
    num_data = len(encoded_list_rearrange_concat[0])
    timestep = 24
    # num_data
    for n in range(num_data):
        print('n: ', n)
        for ds_name1 in all_keys:
            # 1D case
            if ds_name1 in keys_1d:
                temp_arr1 = feature_map_dict[ds_name1][n,:] # (24, 1, 1, 1)
                # (24, 1) - > [32, 20, 24]
                temp_1d_dup = np.repeat(temp_arr1, 32, axis = 1)
                temp_1d_dup = np.repeat(temp_1d_dup, 20, axis = 2)  # 32, 20, 24, 1

                temp_1d_dup = np.squeeze(temp_1d_dup, axis = -1)  #[24, 32, 20,]
                temp_1d_dup = np.moveaxis(temp_1d_dup, 0, -1) # (32, 20, 24)


                dim1 = temp_arr1.shape[0]  # number of layers in the 2d data
        #         dim1 = temp_arr1.shape[-1]  # number of layers in the 2d data
                for ds_name2 in all_keys:
                    # 1D VS 1D
                    if ds_name2 in keys_1d:
                        ave_SR = 0
        #                 print(ds_name1, ds_name2)
                        temp_arr2 = feature_map_dict[ds_name2][n, :]
                        temp_2d_dup = np.repeat(temp_arr2, 32, axis = 1)
                        temp_2d_dup = np.repeat(temp_2d_dup, 20, axis = 2)  # 32, 20, 24, 1
                        temp_2d_dup = np.squeeze(temp_2d_dup, axis = -1)  #[24, 32, 20,]
                        temp_2d_dup = np.moveaxis(temp_2d_dup, 0, -1) # (32, 20, 24)

                        compress_arr2 = remove_outside_cells(temp_2d_dup, mask_arr) # [32, 20, 24]
                        compress_arr1 = remove_outside_cells( temp_1d_dup, mask_arr) # [32, 20, 24]

                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                               compress_arr2.reshape(1, -1))
                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR

                    # 2D VS 1D
                    # 2D:  32, 20, 1
                    # 1D duplicate: 32, 20, 3. This means that there is no spatial variations for 1D
                    # duplicate 2D to 32, 20, 3. This means that there is no temporal variations for 2D
                    # then flatten and compare
                    # This means that there is no temporal variations for 2D
                    if ds_name2 in keys_2d:
                        temp_arr2 = feature_map_dict[ds_name2][n,:,:,:] # 32, 20, 1
                        # duplicate to [32, 20, 24]
                        temp_arr2_mean_dup = np.repeat(temp_arr2, dim1, axis = -1)
                        compress_arr2 = remove_outside_cells(temp_arr2_mean_dup, mask_arr) # [32, 20, 24]
                        compress_arr1 = remove_outside_cells( temp_1d_dup, mask_arr) # [32, 20, 24]

                        ave_SR = 0
                        sim_sparse = cosine_similarity(compress_arr2.reshape(1, -1),
                                                                   compress_arr1.reshape(1, -1))

                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR
                        # relation_all_df.loc[ds_name1, ds_name2]  += 0


                    # 3D VS 1D
                    # duplicate 1D to 3D, flatten and compare
                    if ds_name2 in keys_3d:
                        temp_arr2 = feature_map_dict[ds_name2][n,:,:,:,:] # 3d, e.g. [24, 32, 20, 1]
                        temp_arr2 = np.squeeze(temp_arr2, axis = -1)  #[24, 32, 20]
                        temp_arr2 = np.moveaxis(temp_arr2, 0, -1) # (32, 20, 24)

                        ave_SR = 0

                        compress_arr2 = remove_outside_cells(temp_arr2, mask_arr)
                        compress_arr1 = remove_outside_cells(temp_1d_dup, mask_arr)

                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                                compress_arr2.reshape(1, -1))

                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR


            # 2D case
            if ds_name1 in keys_2d:
                temp_arr1 = feature_map_dict[ds_name1][n,:,:,:]  # [32, 20, 1]
                # duplicate to [32, 20, 24]
                temp_arr1_mean_dup = np.repeat(temp_arr1, timestep, axis = -1)
                for ds_name2 in all_keys:
                    # 2D Vs 1D
                    if ds_name2 in keys_1d:
                        relation_all_df.loc[ds_name1, ds_name2]  = relation_all_df.loc[ds_name2, ds_name1]
                    # 2D Vs 2D
                    # all duplicate to 3D
                    if ds_name2 in keys_2d:
                        ave_SR = 0 # average spearman correlation
        #                 print(ds_name1, ds_name2)
                        temp_arr2 = feature_map_dict[ds_name2][n,:,:,:]
                        temp_arr2_mean_dup = np.repeat(temp_arr2, timestep, axis = -1)

                        compress_arr2 = remove_outside_cells(temp_arr2_mean_dup, mask_arr)
                        compress_arr1 = remove_outside_cells( temp_arr1_mean_dup, mask_arr)

                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                    compress_arr2.reshape(1, -1))
    #                             pearson_coef, p_value = stats.pearsonr(temp_arr1[ :, :, i].ravel(), temp_arr2[ :, :, j].ravel())

                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2] += ave_SR

                    # 2D VS 3D
                    # duplicate 2D to 3D and compare to 3D
                    if ds_name2 in keys_3d:
                        temp_arr2 = feature_map_dict[ds_name2][n,:,:,:,:]     #[24, 32, 20, 1]
                        temp_arr2 = np.squeeze(temp_arr2, axis = -1)  #[24, 32, 20]
                        temp_arr2 = np.moveaxis(temp_arr2, 0, -1) # (32, 20, 24)
                        # temp_arr2 = np.squeeze(temp_arr2, axis = -1)  #[24, 32, 20]
                        # temp_arr2 = np.moveaxis(temp_arr2, 0, -1) # (32, 20, 24)
                        #
                        # # average along third dimension
                        # temp_arr2_mean = np.mean(temp_arr2, axis = -1)
                        # temp_arr2_mean_dup = np.expand_dims(temp_arr2_mean, axis = -1) #[32, 20, 1]


                        compress_arr2 = remove_outside_cells( temp_arr2, mask_arr)
                        compress_arr1 = remove_outside_cells( temp_arr1_mean_dup, mask_arr)

                        ave_SR = 0 # average spearman correlation
                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                   compress_arr2.reshape(1, -1))
                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR

            # 3D
            if ds_name1 in keys_3d:
                temp_arr1 = feature_map_dict[ds_name1][n,:,:,:,:]  # [24, 32, 20, 1]
                temp_arr1 = np.squeeze(temp_arr1, axis = -1)  #[24, 32, 20]
                temp_arr1 = np.moveaxis(temp_arr1, 0, -1) # (32, 20, 24)


                for ds_name2 in all_keys:
                    # 1D
                    if ds_name2 in keys_1d:
                        relation_all_df.loc[ds_name1, ds_name2]  = relation_all_df.loc[ds_name2, ds_name1]
                    # 3D VS 2D
                    if ds_name2 in keys_2d:
                        relation_all_df.loc[ds_name1, ds_name2]  = relation_all_df.loc[ds_name2, ds_name1]

                    # 3D VS 3D
                    # flatten and compare. Because 3rd dimension contains
                    # temporal information
                    if ds_name2 in keys_3d:
                        temp_arr2 = feature_map_dict[ds_name2][n,:,:,:,:]
                        temp_arr2 = np.squeeze(temp_arr2, axis = -1)  #[24, 32, 20]
                        temp_arr2 = np.moveaxis(temp_arr2, 0, -1) # (32, 20, 24)

                        ave_SR = 0 # average spearman correlation
                        compress_arr2 = remove_outside_cells( temp_arr2, mask_arr)
                        compress_arr1 = remove_outside_cells( temp_arr1, mask_arr)

                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                                                       compress_arr2.reshape(1, -1))

                        ave_SR = float(sim_sparse[0][0])
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR

    relation_all_df = relation_all_df / num_data
    return relation_all_df




# 1d feature map shape: [1, 3] -> (24, 1)   originally (24, 1, 1, 1)
# 2d feature map: [32, 20, 1] -> [32, 20, 1]
# 3d feature map: [32, 20, 3]  -> [24, 32, 20, 1]
# all duplicate to 3d and comapre
#grouping within dim
def first_level_grouping_within_group(feature_map_dict, encoded_list_rearrange_concat,
            mask_arr, all_keys,keys_1d, keys_2d, keys_3d =  []):
    height = 32
    width = 20
    relation_1d_df = pd.DataFrame(0, columns = keys_1d, index = keys_1d)
    relation_2d_df = pd.DataFrame(0, columns = keys_2d, index = keys_2d)
    relation_3d_df = pd.DataFrame(0, columns = keys_3d, index = keys_3d)
    num_data = len(encoded_list_rearrange_concat[0])
    timestep = 24
    # num_data
    for n in range(num_data):
        print('n: ', n)
        for ds_name1 in all_keys:
            # 1D case
            if ds_name1 in keys_1d:
                temp_arr1 = feature_map_dict[ds_name1][n,:] # (24, 1, 1, 1)
                # (24, 1) - > [32, 20, 24]
                temp_1d_dup = np.repeat(temp_arr1, 32, axis = 1)
                temp_1d_dup = np.repeat(temp_1d_dup, 20, axis = 2)  # 32, 20, 24, 1

                temp_1d_dup = np.squeeze(temp_1d_dup, axis = -1)  #[24, 32, 20,]
                temp_1d_dup = np.moveaxis(temp_1d_dup, 0, -1) # (32, 20, 24)


                dim1 = temp_arr1.shape[0]  # number of layers in the 2d data
        #         dim1 = temp_arr1.shape[-1]  # number of layers in the 2d data
                for ds_name2 in all_keys:
                    # 1D VS 1D
                    if ds_name2 in keys_1d:
                        temp_arr2 = feature_map_dict[ds_name2][n, :]
                        sim_sparse = cosine_similarity(temp_arr1.reshape(1, -1),
                               temp_arr2.reshape(1, -1))
                        ave_SR = sim_sparse[0][0]
                        relation_1d_df.loc[ds_name1, ds_name2]  += ave_SR
                        # ave_SR = 0
                        #
                        # temp_arr2 = feature_map_dict[ds_name2][n, :]
                        # temp_2d_dup = np.repeat(temp_arr2, 32, axis = 1)
                        # temp_2d_dup = np.repeat(temp_2d_dup, 20, axis = 2)  # 32, 20, 24, 1
                        # temp_2d_dup = np.squeeze(temp_2d_dup, axis = -1)  #[24, 32, 20,]
                        # temp_2d_dup = np.moveaxis(temp_2d_dup, 0, -1) # (32, 20, 24)
                        #
                        # compress_arr2 = remove_outside_cells(temp_2d_dup, mask_arr) # [32, 20, 24]
                        # compress_arr1 = remove_outside_cells( temp_1d_dup, mask_arr) # [32, 20, 24]
                        #
                        # sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                        #        compress_arr2.reshape(1, -1))
                        # ave_SR = sim_sparse[0][0]
                        # relation_1d_df.loc[ds_name1, ds_name2]  += ave_SR

            # 2D case
            if ds_name1 in keys_2d:
                temp_arr1 = feature_map_dict[ds_name1][n,:,:,:]  # [32, 20, 1]
                # duplicate to [32, 20, 24]
                temp_arr1_mean_dup = np.repeat(temp_arr1, timestep, axis = -1)
                for ds_name2 in all_keys:
                    # 2D Vs 2D
                    # all duplicate to 3D
                    if ds_name2 in keys_2d:
        #                 ave_SR = 0 # average spearman correlation
        # #                 print(ds_name1, ds_name2)
        #                 temp_arr2 = feature_map_dict[ds_name2][n,:,:,:]
        #                 temp_arr2_mean_dup = np.repeat(temp_arr2, timestep, axis = -1)
        #
        #                 compress_arr2 = remove_outside_cells(temp_arr2_mean_dup, mask_arr)
        #                 compress_arr1 = remove_outside_cells( temp_arr1_mean_dup, mask_arr)
        #
        #                 sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
        #                             compress_arr2.reshape(1, -1))
        #
        #                 ave_SR = sim_sparse[0][0]
        #                 relation_2d_df.loc[ds_name1, ds_name2] += ave_SR

                        ave_SR = 0 # average spearman correlation
                        temp_arr2 = feature_map_dict[ds_name2][n,:,:,:]

                        compress_arr2 = remove_outside_cells(temp_arr2, mask_arr)
                        compress_arr1 = remove_outside_cells( temp_arr1, mask_arr)

                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                    compress_arr2.reshape(1, -1))

                        ave_SR = sim_sparse[0][0]
                        relation_2d_df.loc[ds_name1, ds_name2] += ave_SR


            # 3D
            if ds_name1 in keys_3d:
                temp_arr1 = feature_map_dict[ds_name1][n,:,:,:,:]  # [24, 32, 20, 1]
                temp_arr1 = np.squeeze(temp_arr1, axis = -1)  #[24, 32, 20]
                temp_arr1 = np.moveaxis(temp_arr1, 0, -1) # (32, 20, 24)


                for ds_name2 in all_keys:

                    # 3D VS 3D
                    # flatten and compare. Because 3rd dimension contains
                    # temporal information
                    if ds_name2 in keys_3d:
                        temp_arr2 = feature_map_dict[ds_name2][n,:,:,:,:]
                        temp_arr2 = np.squeeze(temp_arr2, axis = -1)  #[24, 32, 20]
                        temp_arr2 = np.moveaxis(temp_arr2, 0, -1) # (32, 20, 24)

                        ave_SR = 0 # average spearman correlation
                        compress_arr2 = remove_outside_cells( temp_arr2, mask_arr)
                        compress_arr1 = remove_outside_cells( temp_arr1, mask_arr)

                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                                                       compress_arr2.reshape(1, -1))

                        ave_SR = float(sim_sparse[0][0])
                        relation_3d_df.loc[ds_name1, ds_name2]  += ave_SR

    relation_1d_df = relation_1d_df / num_data
    relation_2d_df = relation_2d_df / num_data
    relation_3d_df = relation_3d_df / num_data
    return relation_1d_df, relation_2d_df, relation_3d_df




# group
def second_level_grouping(feature_map_dict, encoded_list_rearrange_concat,
            mask_arr, all_keys):
    relation_all_df = pd.DataFrame(0, columns = all_keys, index = all_keys)
    num_data = len(encoded_list_rearrange_concat[0])

    for n in range(num_data):
        print('n: ', n)
        for ds_name1 in all_keys:
            temp_arr1 = feature_map_dict[ds_name1]
            # print('temp_arr1.shape: ', temp_arr1.shape)

            for ds_name2 in all_keys:
                temp_arr2 = feature_map_dict[ds_name2]
                #print('temp_arr2[n, :, :, :,:].shape',temp_arr2[n, :, :, :,:].shape, len(temp_arr2[n, :, :, :,:].shape))
                compress_arr2 = remove_outside_cells( temp_arr2[n, :, :, :,:], mask_arr)
                compress_arr1 = remove_outside_cells( temp_arr1[n, :, :, :,:], mask_arr)

                ave_SR = 0
                sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                                            compress_arr2.reshape(1, -1))

                ave_SR = float(sim_sparse[0][0])
                relation_all_df.loc[ds_name1, ds_name2]  += ave_SR

    relation_all_df = relation_all_df / num_data
    return relation_all_df




# default AffinityPropagation
def clustering(relation_all_df, all_keys, txt_name, method = 'AffinityPropagation', n_clusters= 5):
    print('begin clustering')
    data = relation_all_df.iloc[:, :].values
    if method == 'AffinityPropagation':
        clustering = AffinityPropagation(damping=0.8).fit(data)
    if method == 'AgglomerativeClustering':
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        clustering.fit_predict(data)

    res_dict = dict()
    for i in range(len(clustering.labels_)):
        if clustering.labels_[i] not in res_dict:
            res_dict[clustering.labels_[i]] = []
            res_dict[clustering.labels_[i]].append(all_keys[i])
        else:
            res_dict[clustering.labels_[i]].append(all_keys[i])

    for k, v in res_dict.items():
        print(k,v)

    # for key in res_dict.keys():
    #     if type(key) is not str:
    #         res_dict[str(key)] = res_dict[key]
    #         del res_dict[key]

    with open(txt_name, 'w') as the_file:
        # the_file.write(json.dumps(list(res_dict.items())))
        if method == 'AgglomerativeClustering':
            the_file.write('n_clusters: \n')
            the_file.write(str(n_clusters) + '\n')
        for i in res_dict.keys():
            the_file.write(str(i) + '\n')
            the_file.write(','.join([str(x) for x in res_dict[i]]) + "\n")


def plot_grouping(relation_all_df, plot_name):
    data = relation_all_df.iloc[:, :].values
    plt.figure(figsize=(10, 7))
    # plt.title("Customer Dendograms")
    dend = shc.dendrogram(shc.linkage(data, method='ward'),
        labels =relation_all_df.index, orientation = 'left' )
    plt.savefig(plot_name)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',   '--suffix',
                     action="store", help = 'save path suffix', default = '')
    parser.add_argument('-d',   '--encoding_dir',
                     action="store", help = 'dir containing checkpoints and feature maps', default = '')
    parser.add_argument('-l',   '--level',
                     action="store", help = 'Which level to group: first, second, ...', default = 'first')
    parser.add_argument('-m',   '--method',
                     action="store",
                     help = 'clustering method...AgglomerativeClustering, or AffinityPropagation',
                     default = 'AffinityPropagation')
    parser.add_argument('-n',   '--n_clusters',  type=int,
                     action="store", help = 'number of clusters', default = 5)
    parser.add_argument('-g',   '--n_groups',  type=int,
                     action="store", help = 'number of groups to be grouped', default = 9)
    return parser.parse_args()


def main():
    args = parse_args()
    encoding_dir = args.encoding_dir
    level = args.level
    suffix = args.suffix
    method = args.method
    n_clusters = args.n_clusters
    n_groups = args.n_groups
    print("encoding_dir: ", encoding_dir)
    print("level: ", level)

    file = open(join(encoding_dir,'encoded_list'), 'rb')
    # dump information to that file
    encoded_list = pickle.load(file)
    print(len(encoded_list[0]))
    # close the file
    file.close()
    # rearrange encoded_list
    # original dimension: # of batches, # of datasets, [batch_size, ......],
    # arrange into :  # of datasets, # of batches,  [batch_size, ......],

    encoded_list_rearrange = [[None for j in range(len(encoded_list))] for i in range(len(encoded_list[0]))]
    for i, batch in enumerate(encoded_list):
        for j, ds in enumerate(batch):
            encoded_list_rearrange[j][i] = encoded_list[i][j]

    encoded_list_rearrange_concat = [np.concatenate(batch, axis = 0) for batch in encoded_list_rearrange]

    intersect_pos = pd.read_csv('./auxillary_data/intersect_pos_32_20.csv')
    intersect_pos_set = set(intersect_pos['0'].tolist())
    # ----  test removing outside cells ----- #
    mask_arr = generate_mask_array(intersect_pos_set)


    if level == 'first':
        keys_1d = ['precipitation', 'temperature', 'pressure', 'airquality']
        keys_2d = ['house_price', 'POI_business', 'POI_food', 'POI_government',
                   'POI_hospitals', 'POI_publicservices', 'POI_recreation', 'POI_school',
                   'POI_transportation', 'seattle_street',
                   'total_flow_count', 'transit_routes', 'transit_signals', 'transit_stop', 'slope', 'bikelane']
        keys_3d = ['building_permit', 'collisions', 'seattle911calls']
        keys_list = []
        keys_list.extend(keys_1d)
        keys_list.extend(keys_2d)
        keys_list.extend(keys_3d)
        print('key list: ', keys_list)

        feature_map_dict = dict(zip(keys_list, encoded_list_rearrange_concat))

        print('begin grouping')
        relation_all_df = first_level_grouping_simplified(feature_map_dict, encoded_list_rearrange_concat,
                    mask_arr, keys_list, keys_1d, keys_2d, keys_3d)
        # relation_all_df = first_level_grouping(feature_map_dict, encoded_list_rearrange_concat,
        #             mask_arr, keys_list, keys_1d, keys_2d, keys_3d)



        # relation_1d_df, relation_2d_df, relation_3d_df = first_level_grouping_within_group(feature_map_dict, encoded_list_rearrange_concat,
        #      mask_arr, keys_list, keys_1d, keys_2d, keys_3d)


    if level == 'second':
        keys_list = []
        for i in range(1, n_groups+1):
            keys_list.append('group_' + str(i))
        feature_map_dict = dict(zip(keys_list, encoded_list_rearrange_concat))
        relation_all_df = second_level_grouping(feature_map_dict, encoded_list_rearrange_concat,
                    mask_arr, keys_list)

    if level == 'third':
        keys_list = []
        for i in range(1, n_groups+1):
            keys_list.append('group_2_' + str(i))
        feature_map_dict = dict(zip(keys_list, encoded_list_rearrange_concat))
        relation_all_df = second_level_grouping(feature_map_dict, encoded_list_rearrange_concat,
                    mask_arr, keys_list)



    # -------------------  relation_all_df -------------------------- #

    print('relation_all_df')
    print(relation_all_df)
    relation_all_df.to_csv(encoding_dir+  level+  '_level'+ '_grouping_' + suffix + '.csv')

    txt_name = encoding_dir + '_'+ method+'_' +level+  '_level'+ '_grouping_' + suffix + '.txt'
    clustering(relation_all_df, keys_list,txt_name,method, n_clusters)

    print('plotting')
    plot_name = encoding_dir + '_'+ method+'_' +level+  '_level'+ '_grouping_' + suffix + '.png'
    plot_grouping(relation_all_df, plot_name)
    print('plot saved to :', plot_name)


    # -------------------------------------------------------------#


    # ---------------------------   by dim ------------------------- #

    '''
    print('relation_all_df')
    print(relation_1d_df)
    print(relation_2d_df)
    print(relation_3d_df)
    # relation_all_df.to_csv(encoding_dir+  level+  '_level'+ '_grouping_' + suffix + '.csv')

    print('relation_1d_df')
    txt_name = encoding_dir + '_'+ method+'_' +level+  '_level'+ '_grouping_' + suffix + '.txt'
    clustering(relation_1d_df, keys_1d,txt_name,method, n_clusters)


    print('relation_2d_df')
    txt_name = encoding_dir + '_'+ method+'_' +level+  '_level'+ '_grouping_' + suffix + '.txt'
    clustering(relation_2d_df, keys_2d,txt_name,method, n_clusters)

    print('relation_3d_df')
    txt_name = encoding_dir + '_'+ method+'_' +level+  '_level'+ '_grouping_' + suffix + '.txt'
    clustering(relation_3d_df, keys_3d,txt_name,method, n_clusters)
    '''

    # ----------------------------------------------------------- #






if __name__ == '__main__':
    main()
