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

HEIGHT = 32
WIDTH = 20



# if inside city, assign 1.
# if outside city, assign 0
# turn into numpy array mask, True/ False
def generate_mask_array(intersect_pos_set):
    temp_image = [[0 for i in range(HEIGHT)] for j in range(WIDTH)]
    for i in range(HEIGHT):
        for j in range(WIDTH):
            temp_str = str(j)+'_'+str(i)
            if temp_str in intersect_pos_set:
                temp_image[j][i] = 1

    mask_arr = np.array(temp_image)
    mask_arr = np.rot90(mask_arr)
    # print('mask_arr: ', mask_arr)
    # rawdata_arr = np.moveaxis(rawdata_arr, 0, -1)
    # boolean mask
    return mask_arr



# input a tensor [32, 20, n] or [n, 32, 20]
# remove cells outside city, resulting in, e.g. [500, n]
# return a flatten tensor of 500 * n
def remove_outside_cells(tensor, mask_arr):
    demo_mask_arr_expanded = np.expand_dims(mask_arr, 2)  # [1, 2]
            # [1, 32, 20, 1]  -> [1, 1, 32, 20, 1]
            # [1, 32, 20, 1] -> [batchsize, 32, 20, 1]
            # batchsize = tf.shape(prediction)[0]
    demo_mask_arr_expanded = np.tile(demo_mask_arr_expanded, [1,1, tensor.shape[-1]])
    # print('demo_mask_arr_expanded.shape: ', demo_mask_arr_expanded.shape)
    # masked tensor, outside cells should be false / 0
    marr = np.ma.MaskedArray(tensor, mask= demo_mask_arr_expanded)
    # print('masked array: ', marr)

    compressed_arr = np.ma.compressed(marr)

    # print('compressed arr: ', compressed_arr)
    print('compreessed shape: ', compressed_arr.shape)
    return compressed_arr


# TODO:  remove cells outside city
def first_level_grouping(feature_map_dict, encoded_list_rearrange_concat,
            mask_arr, all_keys, keys_1d, keys_2d, keys_3d):
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
                temp_arr1 = feature_map_dict[ds_name1]
                temp_1d_dup = np.repeat(temp_arr1[n,:], 32, axis = 0)
                temp_1d_dup = np.repeat(temp_1d_dup, 20, axis = 1)  # 32, 20, 3
                dim1 = temp_arr1.shape[-1]  # number of layers in the 2d data
        #         dim1 = temp_arr1.shape[-1]  # number of layers in the 2d data
                for ds_name2 in all_keys:
                    # 1D VS 1D
                    if ds_name2 in keys_1d:
                        ave_SR = 0
        #                 print(ds_name1, ds_name2)
                        temp_arr2 = feature_map_dict[ds_name2]
                        sim_sparse = cosine_similarity(temp_arr1[n, :].reshape(1, -1),
                               temp_arr2[n, :].reshape(1, -1))
                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR

                    # 2D VS 1D
                    # duplicate to 3d, flatten and compare
                    if ds_name2 in keys_2d:
                        temp_arr2 = feature_map_dict[ds_name2]
                        temp_arr2_mean = np.mean(temp_arr2[n, :, :, :], axis = -1)  #
                        temp_arr2_mean_dup = np.expand_dims(temp_arr2_mean, axis = -1)
                        # 32, 20, 3
                        temp_arr2_mean_dup = np.repeat(temp_arr2_mean_dup, dim1, axis = -1)

                        compress_arr2 = remove_outside_cells(temp_arr2_mean_dup, mask_arr)
                        compress_arr1 = remove_outside_cells( temp_1d_dup, mask_arr)

                        ave_SR = 0 # average spearman correlation
                        sim_sparse = cosine_similarity(compress_arr2.reshape(1, -1),
                                                                   compress_arr1.reshape(1, -1))

                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR

                    # 3D VS 1D
                    if ds_name2 in keys_3d:
                        temp_arr2 = feature_map_dict[ds_name2] # 3d, e.g. (32, 20, 3)
                        #temp_1d_dup = np.moveaxis(temp_1d_dup,0, -1) # (32, 20, 3)
                        print('temp_1d_dup.shape ', temp_1d_dup.shape)
                        ave_SR = 0 # average spearman correlation

                        compress_arr2 = remove_outside_cells(temp_arr2[n,:,:,:], mask_arr)
                        compress_arr1 = remove_outside_cells(temp_1d_dup, mask_arr)

                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                                compress_arr2.reshape(1, -1))

                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR


            # 2D case
            if ds_name1 in keys_2d:
                temp_arr1 = feature_map_dict[ds_name1]
                dim1 = temp_arr1.shape[-1]  # number of layers in the 2d data
                temp_arr1_mean = np.mean(temp_arr1[n, :, :, :], axis = -1)  #[32, 20]
                # print('temp_arr1_mean.shape: ', temp_arr1_mean.shape)
                temp_arr1_mean_dup = np.expand_dims(temp_arr1_mean, axis = -1) #[32, 20, 1]
                # temp_arr1_mean_dup = np.repeat(temp_arr1_mean_dup, temp_arr2.shape[-1], axis = 0)

                for ds_name2 in all_keys:
                    # 2D Vs 1D
                    if ds_name2 in keys_1d:
                        relation_all_df.loc[ds_name1, ds_name2]  = relation_all_df.loc[ds_name2, ds_name1]
                    # 2D Vs 2D
                    # take mean along 3rd dimension and compare
                    if ds_name2 in keys_2d:
                        ave_SR = 0 # average spearman correlation
        #                 print(ds_name1, ds_name2)
                        temp_arr2 = feature_map_dict[ds_name2]
                        dim2 = temp_arr2.shape[-1]
                        temp_arr2_mean = np.mean(temp_arr2[n, :, :, :], axis = -1)
                        temp_arr2_mean_dup = np.expand_dims(temp_arr2_mean, axis = -1) #[32, 20, 1]

                        compress_arr2 = remove_outside_cells(temp_arr2_mean_dup, mask_arr)
                        compress_arr1 = remove_outside_cells( temp_arr1_mean_dup, mask_arr)

                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                    compress_arr2.reshape(1, -1))
    #                             pearson_coef, p_value = stats.pearsonr(temp_arr1[ :, :, i].ravel(), temp_arr2[ :, :, j].ravel())

                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2] += ave_SR

                    # 2D VS 3D
                    # for 2D feature maps, 3rd dimension is 3.
                    # flatten and compare
                    if ds_name2 in keys_3d:
                        temp_arr2 = feature_map_dict[ds_name2]

                        compress_arr2 = remove_outside_cells( temp_arr2[n, :, :, :], mask_arr)
                        compress_arr1 = remove_outside_cells( temp_arr1, mask_arr)

                        ave_SR = 0 # average spearman correlation
                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                   compress_arr2.reshape(1, -1))
                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR

            # 3D
            if ds_name1 in keys_3d:
                temp_arr1 = feature_map_dict[ds_name1]
                for ds_name2 in all_keys:
                    # 1D
                    if ds_name2 in keys_1d:
                        relation_all_df.loc[ds_name1, ds_name2]  = relation_all_df.loc[ds_name2, ds_name1]
                    # 3D VS 2D
                    if ds_name2 in keys_2d:
                        temp_arr2 = feature_map_dict[ds_name2]

                        relation_all_df.loc[ds_name1, ds_name2]  += relation_all_df.loc[ds_name2, ds_name1]

                    # 3D VS 3D
                    if ds_name2 in keys_3d:
                        temp_arr2 = feature_map_dict[ds_name2]
                        ave_SR = 0 # average spearman correlation
    #                     for i in range(dim2):
                        compress_arr2 = remove_outside_cells( temp_arr2[n, :, :, :], mask_arr)
                        compress_arr1 = remove_outside_cells( temp_arr1[n, :, :, :], mask_arr)

                        sim_sparse = cosine_similarity(compress_arr1.reshape(1, -1),
                                                                       compress_arr2.reshape(1, -1))

                        ave_SR = float(sim_sparse[0][0])
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR
    relation_all_df = relation_all_df / num_data
    return relation_all_df



def clustering(relation_all_df, all_keys,txt_name):
    print('begin clustering')
    data = relation_all_df.iloc[:, :].values
    clustering = AffinityPropagation(damping=0.8).fit(data)
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
        for i in res_dict.keys():
            the_file.write(str(i) + '\n')
            the_file.write(','.join([str(x) for x in res_dict[i]]) + "\n")




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',   '--suffix',
                     action="store", help = 'save path suffix', default = '')
    parser.add_argument('-d',   '--encoding_dir',
                     action="store", help = 'dir containing checkpoints and feature maps', default = '')
    parser.add_argument('-l',   '--level',
                     action="store", help = 'Which level to group: first, second, ...', default = 'First')
    return parser.parse_args()


def main():
    args = parse_args()
    encoding_dir = args.encoding_dir
    level = args.level
    suffix = args.suffix
    print("encoding_dir: ", encoding_dir)
    print("levelh: ", level)

    keys_1d = ['weather', 'airquality']
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

    feature_map_dict = dict(zip(keys_list, encoded_list_rearrange_concat))
    # encoded_list: (num_dataset, # of data points, 32, 20, dim)

    intersect_pos = pd.read_csv('./auxillary_data/intersect_pos_32_20.csv')
    intersect_pos_set = set(intersect_pos['0'].tolist())

    # ----  test removing outside cells ----- #
    print(encoded_list_rearrange_concat[18][0].shape)
    test_tensor = encoded_list_rearrange_concat[18][0]  # should be [ 32, 20, dim]
    mask_arr = generate_mask_array(intersect_pos_set)

    # compressed_arr = remove_outside_cells(test_tensor, mask_arr)

    print('begin grouping')
    relation_all_df = first_level_grouping(feature_map_dict, encoded_list_rearrange_concat,
                mask_arr, keys_list, keys_1d, keys_2d, keys_3d)
    txt_name = encoding_dir +  level+  '_level'+ '_grouping_' + suffix + '.txt'

    clustering(relation_all_df, keys_list,txt_name)




if __name__ == '__main__':
    main()
