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


# TODO:  remove cells outside city
def first_level_grouping(feature_map_dict, encoded_list_rearrange_concat, all_keys, keys_1d, keys_2d, keys_3d):
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
                        temp_arr2_mean_dup = np.expand_dims(temp_arr2_mean, axis = 0)
                        # 3, 32, 20
                        temp_arr2_mean_dup = np.repeat(temp_arr2_mean_dup, dim1, axis = 0)

                        ave_SR = 0 # average spearman correlation
                        sim_sparse = cosine_similarity(temp_arr2_mean_dup.reshape(1, -1),
                                                                   temp_1d_dup.reshape(1, -1))

                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR

                    # 3D VS 1D
                    if ds_name2 in keys_3d:
                        temp_arr2 = feature_map_dict[ds_name2] # 3d, e.g. (32, 20, 3)
                        temp_1d_dup = np.moveaxis(temp_1d_dup,0, -1) # (32, 20, 3)
                        ave_SR = 0 # average spearman correlation

                        sim_sparse = cosine_similarity(temp_1d_dup.reshape(1, -1),
                                                temp_arr2[n,:,:,:].reshape(1, -1))

                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR


            # 2D case
            if ds_name1 in keys_2d:
                temp_arr1 = feature_map_dict[ds_name1]
                dim1 = temp_arr1.shape[-1]  # number of layers in the 2d data
                temp_arr1_mean = np.mean(temp_arr1[n, :, :, :], axis = -1)
                temp_arr1_mean_dup = np.expand_dims(temp_arr1_mean, axis = 0)
                temp_arr1_mean_dup = np.repeat(temp_arr1_mean_dup, temp_arr2.shape[-1], axis = 0)

                for ds_name2 in all_keys:
                    # 2D Vs 1D
                    if ds_name2 in keys_1d:
                        relation_all_df.loc[ds_name1, ds_name2]  = relation_all_df.loc[ds_name2, ds_name1]
                    # 2D Vs 2D
                    if ds_name2 in keys_2d:
                        ave_SR = 0 # average spearman correlation
        #                 print(ds_name1, ds_name2)
                        temp_arr2 = feature_map_dict[ds_name2]
                        dim2 = temp_arr2.shape[-1]
                        temp_arr2_mean = np.mean(temp_arr2[n, :, :, :], axis = -1)
                        sim_sparse = cosine_similarity(temp_arr1_mean.ravel().reshape(1, -1),
                                    temp_arr2_mean.ravel().reshape(1, -1))
    #                             pearson_coef, p_value = stats.pearsonr(temp_arr1[ :, :, i].ravel(), temp_arr2[ :, :, j].ravel())

                        ave_SR = sim_sparse[0][0]
                        relation_all_df.loc[ds_name1, ds_name2] += ave_SR

                    # 2D VS 3D
                    if ds_name2 in keys_3d:
                        temp_arr2 = feature_map_dict[ds_name2]

                        ave_SR = 0 # average spearman correlation
                        sim_sparse = cosine_similarity(temp_arr1_mean_dup.ravel().reshape(1, -1),
                                   temp_arr2[n, :, :, :].ravel().reshape(1, -1))
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
                        sim_sparse = cosine_similarity(temp_arr1[n,  :,:, :].reshape(1, -1),
                                                                       temp_arr2[n, :,:, :].reshape(1, -1))

                        ave_SR = float(sim_sparse[0][0])
                        relation_all_df.loc[ds_name1, ds_name2]  += ave_SR
    relation_all_df = relation_all_df / num_data
    return relation_all_df



def clustering(relation_all_df, encoding_dir):
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

    txt_name = encoding_dir + '_'+ level+  '_level'+ '_grouping_' + timer + '.txt'
    with open(txt_name, 'w') as the_file:
        the_file.write(json.dumps(res_dict))




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',   '--encoding_dir',
                     action="store", help = 'dir containing checkpoints and feature maps', default = '')
    parser.add_argument('-l',   '--level',
                     action="store", help = 'Which level to group: first, second, ...', default = 'First')
    return parser.parse_args()


def main():
    args = parse_args()
    encoding_dir = args.encoding_dir
    level = args.level
    print("encoding_dir: ", encoding_dir)
    print("levelh: ", level)

    keys_1d = ['weather', 'airquality']
    keys_2d = ['house_price', 'POI_business', 'POI_food', 'POI_government',
               'POI_hospitals', 'POI_publicservices', 'POI_recreation', 'POI_school',
               'POI_transportation', 'seattle_street',
               'total_flow_count', 'transit_routes', 'transit_signals', 'transit_stop', 'slope', 'bikelane']
    keys_3d = ['building_permit', 'collisions', 'seattle911calls']
    keys_list = []
    keys_list.extend(rawdata_1d_dict.keys())
    keys_list.extend(rawdata_2d_dict.keys())
    keys_list.extend(rawdata_3d_dict.keys())
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

    print('begin grouping')
    relation_all_df = first_level_grouping(feature_map_dict, encoded_list_rearrange_concat,
                keys_list, keys_1d, keys_2d, keys_3d)

    clustering(relation_all_df, encoding_dir)
