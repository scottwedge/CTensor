'''
Create noising data for denoising AE
- corrput about 15% of the data, filling with -1
'''

import pandas as pd
import numpy as np
import json
import sys

import glob # read multiple files
import os
import os.path
from os import getcwd
from os.path import join
from os.path import basename   # get file name
import collections
import matplotlib.pyplot as plt
import argparse


import time
import datetime
from datetime import timedelta
import datetime_utils
from matplotlib import pyplot as plt
import random
import pickle



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


# size : (45984, 3)
def corrupt_1d_data_with_neg(input_arr):
    total_len = input_arr.shape[0]
    dim = input_arr.shape[1]
    sample_size = int(total_len * 0.15)
    new_arr = []

    for d in range(dim):
        input_arr_copy = np.copy(input_arr[:, d])
        # make temp mask from sample locations
        mask=np.array([1]*total_len)
        inds=np.random.choice(np.arange(total_len),size=sample_size)
        for i in range(inds.shape[0]):
            input_arr_copy[inds[i]] = -1
#         mask[inds]=0
#         new_arr.append(input_arr[:, i] * mask)
#         new_arr.append(input_arr[:, i] * mask)
        new_arr.append(input_arr_copy)

    new_arr = np.array(new_arr).reshape((dim, total_len))
    new_arr = np.swapaxes(new_arr,0,1)
    return new_arr


# input (32, 20, 1)
# set masked data to be -1
def corrupt_2d_data_with_neg(input_arr, mask_arr):
    # # remove cells outside first
    dim = input_arr.shape[-1]
    new_arr = []
    inside_coor = []
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if mask_arr[i, j ] ==1:
                inside_coor.append((j, i))

    for d in range(dim):
        # make temp mask from sample locations
        inds=np.random.choice(np.arange(len(inside_coor)),size= int(len(inside_coor) * 0.15))
        chosen_idx = np.array(inside_coor)[inds]  # sampled coordinates
        input_arr_copy = np.copy(input_arr[:, :, d])
        # set 0 at sampled location
        for j in range(chosen_idx.shape[0]):
            # mask[ chosen_idx[j, 1], chosen_idx[j, 0]] = False
            input_arr_copy[chosen_idx[j, 1], chosen_idx[j, 0]]  = -1

        new_arr.append(input_arr_copy)

    new_arr = np.array(new_arr).reshape((input_arr.shape[0], input_arr.shape[1], dim))
#     new_arr = np.swapaxes(new_arr,0,1)
    return new_arr



# coordinates inside city
# inside_coor
# input (45984, 32, 20)
def corrupt_3_data_with_neg(input_arr, mask_arr):
    # # remove cells outside first
    dim = input_arr.shape[0]  # 45984
    new_arr = []
    inside_coor = []
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if mask_arr[i, j ] ==1:
                inside_coor.append((j, i))

    for d in range(dim):
        print(d)
        # make temp mask from sample locations
        inds=np.random.choice(np.arange(len(inside_coor)),size= int(len(inside_coor) * 0.15))
        chosen_idx = np.array(inside_coor)[inds]  # sampled coordinates
#         mask=np.array([[1 for i in range(WIDTH)] for j in range(HEIGHT)])
        input_arr_copy = np.copy(input_arr[d, :, :])
        # set 0 at sampled location
        for j in range(chosen_idx.shape[0]):
            # mask[ chosen_idx[j, 1], chosen_idx[j, 0]] = 0
            input_arr_copy[chosen_idx[j, 1], chosen_idx[j, 0]]  = -1

        new_arr.append(input_arr_copy)

    new_arr = np.array(new_arr).reshape((dim, input_arr.shape[1], input_arr.shape[2]))
    return new_arr



def main():
    intersect_pos = pd.read_csv('../auxillary_data/intersect_pos_32_20.csv')
    intersect_pos_set = set(intersect_pos['0'].tolist())
    mask_arr = generate_mask_array(intersect_pos_set)

    # ---- reading data ---------------------#
    print('Reading 1d, 2d, and 3d data')
    path_1d = '../data_processing/1d_source_data/'
    path_2d = '../data_processing/2d_source_data/'
    path_3d = '../data_processing/3d_source_data/'
    # 1d
    weather_arr = np.load(path_1d + 'weather_arr_20140201_20190501.npy')
    airquality_arr = np.load(path_1d + 'air_quality_arr_20140201_20190501.npy')
    print('weather_arr.shape: ', weather_arr.shape)
    print('airquality_arr.shape: ', airquality_arr.shape)
    weather_arr = weather_arr[0,0,:,:]
    airquality_arr = airquality_arr[0,0,:,:]


        # 2d
    house_price_arr = np.load(path_2d + 'house_price.npy')
    POI_business_arr = np.load(path_2d + 'POI_business.npy')
    POI_food_arr = np.load(path_2d + 'POI_food.npy')
    POI_government_arr = np.load(path_2d + 'POI_government.npy')
    POI_hospitals_arr = np.load(path_2d + 'POI_hospitals.npy')
    POI_publicservices_arr = np.load(path_2d + 'POI_publicservices.npy')

    POI_recreation_arr = np.load(path_2d + 'POI_recreation.npy')
    POI_school_arr = np.load(path_2d + 'POI_school.npy')
    POI_transportation_arr = np.load(path_2d + 'POI_transportation.npy')
    seattle_street_arr = np.load(path_2d + 'seattle_street.npy')
    total_flow_count_arr = np.load(path_2d + 'total_flow_count.npy')
    transit_routes_arr = np.load(path_2d + 'transit_routes.npy')
    transit_signals_arr = np.load(path_2d + 'transit_signals.npy')
    transit_stop_arr = np.load(path_2d + 'transit_stop.npy')

    slope_arr = np.load(path_2d + 'slope_arr.npy')
    bikelane_arr = np.load(path_2d + 'bikelane_arr.npy')

    print('transit_routes_arr.shape: ', transit_routes_arr.shape)
    print('POI_recreation_arr.shape: ', POI_recreation_arr.shape)

    # 3d
    building_permit_arr = np.load(path_3d + 'building_permit_arr_20140201_20190501_python3.npy')
    collisions_arr = np.load(path_3d + 'collisions_arr_20140201_20190501_python3.npy')
    crime_arr = np.load(path_3d + 'crime_arr_20140201_20190501_python3.npy')
    seattle911calls_arr = np.load(path_3d + 'seattle911calls_arr_20140201_20190501.npy')
    print('building_permit_arr.shape:', building_permit_arr.shape)
    print('collisions_arr.shape: ', collisions_arr.shape)
    print('crime_arr.shape: ', crime_arr.shape)
    print('seattle911calls_arr.shape: ', seattle911calls_arr.shape)

    # duplicate building_permit_arr and collisions to the same shape as seattle911calls
    # deal them the same way as 911
    building_permit_arr_seq_extend = np.repeat(building_permit_arr, 24, axis =0)
    collisions_arr_seq_extend = np.repeat(collisions_arr, 24, axis =0)

    print('building_permit_arr_seq_extend.shape: ', building_permit_arr_seq_extend.shape)

    # construct dictionary
    print('use dictionary to organize data')

    rawdata_1d_dict = {
     'precipitation':  np.expand_dims(weather_arr[:,0], axis=1) ,
    'temperature':  np.expand_dims(weather_arr[:,1], axis=1) ,
    'pressure':  np.expand_dims(weather_arr[:,2], axis=1),
    'airquality': airquality_arr,
    }

    rawdata_2d_dict = {
        'house_price': house_price_arr,
        'POI_business': POI_business_arr,
        'POI_food': POI_food_arr,
        'POI_government': POI_government_arr,
        'POI_hospitals': POI_hospitals_arr,
        'POI_publicservices': POI_publicservices_arr,
        'POI_recreation': POI_recreation_arr,
        'POI_school': POI_school_arr,
        'POI_transportation': POI_transportation_arr,
        'seattle_street': seattle_street_arr,
        'total_flow_count': total_flow_count_arr,
        'transit_routes': transit_routes_arr,
        'transit_signals': transit_signals_arr,
        'transit_stop':transit_stop_arr,
        'slope': slope_arr,
        'bikelane': bikelane_arr,
        }

    rawdata_3d_dict = {
          'building_permit': building_permit_arr_seq_extend,
        'collisions': collisions_arr_seq_extend,  # expect (1, 45984, 32, 20)
        # 'building_permit': building_permit_arr,
        # 'collisions':collisions_arr,
        'seattle911calls': seattle911calls_arr # (45984, 32, 20)
        }

    keys_1d = list(rawdata_1d_dict.keys())
    keys_2d = list(rawdata_2d_dict.keys())
    keys_3d = list(rawdata_3d_dict.keys())

    #------------------  Corruption ---------------------------- #
    rawdata_1d_corrupted_dict = {}
    for k, v in rawdata_1d_dict.iteritems():
        print('creating data for ', k)
        corrupted_v = corrupt_1d_data_with_neg(v)
        rawdata_1d_corrupted_dict[k] = corrupted_v


    rawdata_2d_corrupted_dict = {}
    for k, v in rawdata_2d_dict.iteritems():
        print('creating data for ', k)
        corrupted_v = corrupt_2d_data_with_neg(v)
        rawdata_2d_corrupted_dict[k] = corrupted_v


    rawdata_3d_corrupted_dict = {}
    for k, v in rawdata_3d_dict.iteritems():
        print('creating data for ', k)
        corrupted_v = corrupt_3d_data_with_neg(v)
        rawdata_3d_corrupted_dict[k] = corrupted_v

    # save the ditionaries
    rawdata_1d_corrupted_file = open(path_1d + 'rawdata_1d_corrupted_dict', 'wb')
    print('saved dict to : ',path_1d + 'rawdata_1d_corrupted_dict')
    pickle.dump(rawdata_1d_corrupted_dict, rawdata_1d_corrupted_file)
    rawdata_1d_corrupted_file.close()

    rawdata_2d_corrupted_file = open(path_2d + 'rawdata_2d_corrupted_dict', 'wb')
    print('saved dict to : ',path_2d + 'rawdata_2d_corrupted_dict')
    pickle.dump(rawdata_2d_corrupted_dict, rawdata_2d_corrupted_file)
    rawdata_2d_corrupted_file.close()


    rawdata_3d_corrupted_file = open(path_3d + 'rawdata_3d_corrupted_dict', 'wb')
    print('saved dict to : ',path_3d + 'rawdata_3d_corrupted_dict')
    pickle.dump(rawdata_3d_corrupted_dict, rawdata_3d_corrupted_file)
    rawdata_3d_corrupted_file.close()



if __name__ == '__main__':
    main()
