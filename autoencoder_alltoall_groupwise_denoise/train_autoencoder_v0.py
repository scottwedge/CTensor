# updated April 26
# train individual AE for each group
# grouping is determined according to semantic / dim / similarities, etc.
# groups are not neccessairly mutually exclusive.

'''
Semantic:
grouping_dict = {
'weather_grp': ['precipitation','temperature', 'pressure', 'airquality'],
'transportation_grp': ['POI_transportation', 'seattle_street', 'total_flow_count',
                 'transit_routes', 'transit_signals', 'transit_stop', 'bikelane',
                  'collisions', 'slope'],
'economics_grp': ['house_price', 'POI_business', 'POI_food', 'building_permit',
                         'seattle911calls'],
'public_service_grp': ['POI_government', 'POI_hospitals', 'POI_publicservices',
                 'POI_recreation', 'POI_school']
            }
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
#import lstm
import evaluation
import autoencoder_v0
from matplotlib import pyplot as plt
import random
import pickle



HEIGHT = 32
WIDTH = 20
TIMESTEPS = 24

CHANNEL = 27  # number of all features

BATCH_SIZE = 32
# actually epochs
# TRAINING_STEPS = 200
TRAINING_STEPS = 50

LEARNING_RATE = 0.001

# HOURLY_TIMESTEPS = 168

HOURLY_TIMESTEPS = 24
DAILY_TIMESTEPS = 7
THREE_HOUR_TIMESTEP = 56

# stacking to form features: [9504-168, 168, 32, 20, 9]
# target latent representation: [9504-168, 1, 32,20,1]
def generate_fixlen_timeseries(rawdata_arr, timestep = 24):
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


fea_list = ['pop','normalized_pop', 'bi_caucasian','bi_age', 'bi_high_incm','bi_edu_univ', 'bi_nocar_hh',
           'white_pop','age65_under', 'edu_uni']

'''
train_start_time = '2017-10-01',train_end_time = '2018-08-31',
test_start_time = '2018-09-01 00:00:00', test_end_time = '2018-10-31 23:00:00'
'''
class train:
    # 63 months in total , 57 months for training, 6 months for testing
    def __init__(self,  demo_raw,
                train_start_time = '2014-02-01',train_end_time = '2018-10-31',
                test_start_time = '2018-11-01 00:00:00', test_end_time = '2019-05-01 23:00:00' ):
        # self.raw_df = raw_df
        # demongraphic data [32, 32, 14]
        self.demo_raw = demo_raw
        self.train_start_time = train_start_time
        #self.train_end_time = '2018-03-31'
        self.train_end_time = train_end_time
        # set train/test set
        self.test_start_time = test_start_time
        self.test_end_time = test_end_time
        # prediction window: use one week's data to predict next hour
        self.window = datetime.timedelta(hours=24 * 7)
        self.step = datetime.timedelta(hours=1)
        # predict_start_time should be '2018-04-08 00:00:00'
        # e.g. use '2018-04-01 00:00:00' -> '2018-04-07 23:00:00', in total 168 time stamps
        # to predict  '2018-04-08 00:00:00'
        # however, test_start_time + window = predict_start_time
        # e.g. '2018-04-01 00:00:00'  + 168 hour window = '2018-04-08 00:00:00'
        # this is calculated by time interval, there is 1 hour shift between timestamp and time interval
        self.predict_start_time = datetime_utils.str_to_datetime(self.test_start_time) + self.window
        # predict_end_time = test_end_time = '2018-04-30 23:00:00'
        self.predict_end_time = datetime_utils.str_to_datetime(self.test_end_time)
        # if window = 7 days, test_end_time  = '2018-04-30 23:00:00', actual_end_time =  04/23 - 23:00
        self.actual_end_time = self.predict_end_time - self.window
        # 41616
        self.train_hours = datetime_utils.get_total_hour_range(self.train_start_time, self.train_end_time)




    # TODO: use white, not hispanic, instead of white alone for Austin
    # the feature name is white_nonh, not white
    def generate_binary_demo_attr(self, intersect_pos_set,
            bi_caucasian_th = 65.7384, age65_th = 13.01,
            hh_incm_hi_th = 41.76, edu_uni_th =53.48, no_car_hh_th = 16.94 ):
        # outside city boundary is 0
        self.demo_raw['bi_caucasian'] = [0]*len(self.demo_raw)
        self.demo_raw['bi_age'] = [0]*len(self.demo_raw)
        self.demo_raw['bi_high_incm'] = [0]*len(self.demo_raw)
        self.demo_raw['bi_edu_univ'] = [0]*len(self.demo_raw)
        self.demo_raw['bi_nocar_hh'] = [0]*len(self.demo_raw)

        self.demo_raw['mask'] = [0]*len(self.demo_raw)


        # should ignore cells that have no demo features
        for idx, row in self.demo_raw.iterrows():
            #print('idx: ', idx)
            if row['pos'] not in intersect_pos_set:
                continue
            # added, exclude grids that 0 population from training
            # if row['pop'] != 0:
            self.demo_raw.loc[idx,'mask'] = 1
            # caucasian = 1
            # if row['asian_pop'] + row['black_pop'] + row['hispanic_p'] < 34.27:
            #     self.demo_raw.loc[idx,'bi_caucasian'] = 1
            # else:
            #     self.demo_raw.loc[idx,'bi_caucasian'] = -1
            if row['white_pop'] >= bi_caucasian_th:
                self.demo_raw.loc[idx,'bi_caucasian'] = 1
            else:
                self.demo_raw.loc[idx,'bi_caucasian'] = -1

            # young = 1
            if row['age65'] < age65_th:
                self.demo_raw.loc[idx,'bi_age'] = 1
            else:
                self.demo_raw.loc[idx,'bi_age'] = -1
            # high_income = 1
            if row['hh_incm_hi'] > hh_incm_hi_th:
                self.demo_raw.loc[idx,'bi_high_incm'] = 1
            else:
                self.demo_raw.loc[idx,'bi_high_incm'] = -1
            # edu_univ = 1
            if row['edu_uni'] > edu_uni_th:
                self.demo_raw.loc[idx,'bi_edu_univ'] = 1
            else:
                self.demo_raw.loc[idx,'bi_edu_univ'] = -1
            # more car = 1
            if row['no_car_hh'] < no_car_hh_th:
                self.demo_raw.loc[idx,'bi_nocar_hh'] = 1
            else:
                self.demo_raw.loc[idx,'bi_nocar_hh'] = -1
        self.demo_raw['normalized_pop'] =  self.demo_raw['pop'] / self.demo_raw['pop'].sum()
        # added for metric 2
        self.demo_raw['age65_under'] = 100- self.demo_raw['age65']


    # make mask for demo data
    def demo_mask(self):
    #     if demo_arr is None:
    #         raw_df = demo_raw.fillna(0)

    #         raw_df = demo_arr.fillna(0)
        rawdata_list = list()
        # add a dummy col
        temp_image = [[0 for i in range(HEIGHT)] for j in range(WIDTH)]
        series = self.demo_raw['mask']
        for i in range(len(self.demo_raw)):
            r = self.demo_raw['row'][i]
            c = self.demo_raw['col'][i]
            temp_image[r][c] = series[i]
            temp_arr = np.array(temp_image)
            temp_arr = np.rot90(temp_arr)
        rawdata_list.append(temp_arr)

        rawdata_arr = np.array(rawdata_list)
            # move axis -> [32, 32, 14]
        rawdata_arr = np.moveaxis(rawdata_arr, 0, -1)
        return rawdata_arr  # mask_arr



    '''
    input_df:
                 region_code1, region_code2, ....
    timestamp1
    timestamp2
    ....

    return: array [timestamp, width, height]
            e.g. [10000, 30, 30]
    '''
    def df_to_tensor(self):
        rawdata_list = list()
        for idx, dfrow in self.raw_df.iterrows():
            #('idx: ', idx)
            # an image is list of list: [01,02,....029], [10,11, ....]
            temp_image = [[0 for i in range(HEIGHT)] for j in range(WIDTH)]
            for col in list(self.raw_df ):
                # 9_28: r  = 9,   c = 28
                r = int(col.split('_')[0])
                c = int(col.split('_')[1])
                temp_image[r][c] = dfrow[col]
                temp_arr = np.array(temp_image)
                temp_arr = np.rot90(temp_arr)
            rawdata_list.append(temp_arr)
        rawdata_arr = np.array(rawdata_list)
        return rawdata_arr



    # demographic data to array: [32, 32, 14]
    def demodata_to_tensor(self, demo_arr = None):
        if demo_arr is None:
            raw_df = self.demo_raw.fillna(0)

        raw_df = demo_arr.fillna(0)
        # [len(fea) , 32, 32]
        rawdata_list = list()
        for fea in fea_list:
            temp_image = [[0 for i in range(HEIGHT)] for j in range(WIDTH)]
            series = raw_df[fea]
            for i in range(len(raw_df)):
                r = raw_df['row'][i]
                c = raw_df['col'][i]
                temp_image[r][c] = series[i]
                temp_arr = np.array(temp_image)
                temp_arr = np.rot90(temp_arr)
            rawdata_list.append(temp_arr)
        # [14, 32, 32]
        rawdata_arr = np.array(rawdata_list)
        # move axis -> [32, 32, 14]
        rawdata_arr = np.moveaxis(rawdata_arr, 0, -1)
        return rawdata_arr


    # transform demographic data to tensor
    # with selected features to be used in prediction
    # normalize the features to [0,1]
    def selected_demo_to_tensor(self):
        fea_to_include = fea_list.copy()
        fea_to_include.extend(['pos', 'row','col'])

        selected_demo_df = self.demo_raw[fea_to_include]
        # for fea in fea_list:
        #     selected_demo_df[fea] = selected_demo_df[fea] / selected_demo_df[fea].max()

        demo_arr = self.demodata_to_tensor(selected_demo_df)
        return demo_arr


    # generate time series
    '''
    input: rawdata_arr [timestamp, width, height]
    return: [ (moving window length + horizon len), # of examples, width, height]
             e.g. [(169, # of examples, 30, 30)]
    '''
    ################################
    #  note when generating sequecnces, generate [(168, # of examples, 30, 30)]
    # instead of  [(169, # of examples, 30, 30)]
    # as the latent representation for a week.
    ################################
    def generate_fixlen_timeseries(self, rawdata_arr):
        raw_seq_list = list()
        # arr_shape: [# of timestamps, w, h]
        arr_shape = rawdata_arr.shape
        #for i in range(0, arr_shape[0] - (TIMESTEPS + 1)+1):
        for i in range(0, arr_shape[0] - (TIMESTEPS)+1):
            start = i
            end = i+ (TIMESTEPS )
            # temp_seq = rawdata_arr[start: end, :, :]
            temp_seq = rawdata_arr[start: end]
            raw_seq_list.append(temp_seq)
        raw_seq_arr = np.array(raw_seq_list)
        raw_seq_arr = np.swapaxes(raw_seq_arr,0,1)
        return raw_seq_arr



    # split train/test according to predefined timestamps
    '''
    return:
        train_arr: e.g.:[(169, # of training examples, 30, 30)]
    '''
    def train_test_split(self,raw_seq_arr):
        train_hours = datetime_utils.get_total_hour_range(self.train_start_time, self.train_end_time)
        # train_arr = raw_seq_arr[:, :train_hours, :, :]
        # test_arr = raw_seq_arr[:, train_hours:, :, :]
        train_arr = raw_seq_arr[:, :train_hours]
        test_arr = raw_seq_arr[:, train_hours:]
        return train_arr, test_arr





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',   '--suffix',
                     action="store", help = 'save path suffix', default = '')
    parser.add_argument('-k',   '--key',
                     action="store", help = 'train only one dataset', default = '')
    parser.add_argument('-d',   '--dim',  type=int,
                     action="store", help = 'dims of latent rep', default = 3)
    parser.add_argument("-r","--resume_training", type=bool, default=False,
    				help="A boolean value whether or not to resume training from checkpoint")
    parser.add_argument('-t',   '--train_dir',
                     action="store", help = 'training dir containing checkpoints', default = '')
    parser.add_argument('-c',   '--checkpoint',
                     action="store", help = 'checkpoint path (resume training)', default = None)
    parser.add_argument('-p',   '--place',
                     action="store", help = 'city to train on: Seattle or Austin', default = 'Seattle')
    parser.add_argument('-e',   '--epoch',  type=int,
                     action="store", help = 'epochs to train', default = 50)
    parser.add_argument('-l',   '--learning_rate',  type=float,
                     action="store", help = 'epochs to train', default = 0.01)
    parser.add_argument("-i","--inference", type=bool, default=False,
    				help="inference")
    parser.add_argument("-up","--use_pretrained", type=bool, default=False,
        				help="A boolean value whether or not to start from pretrained model")
    parser.add_argument('-pc',   '--pretrained_checkpoint',
                         action="store", help = 'checkpoint path to pretrained models', default = None)



    return parser.parse_args()



def main():
    args = parse_args()
    suffix = args.suffix
    # the following arguments for resuming training
    resume_training = args.resume_training
    train_dir = args.train_dir
    checkpoint = args.checkpoint
    place = args.place
    epoch = args.epoch
    learning_rate= args.learning_rate
    dim = args.dim
    inference = args.inference
    key = args.key
    use_pretrained = args.use_pretrained
    # this is a path to a list of individually trained checkpoints
    pretrained_checkpoint = args.pretrained_checkpoint

    print("resume_training: ", resume_training)
    print("training dir path: ", train_dir)
    print("checkpoint: ", checkpoint)
    print("place: ", place)
    print("epochs to train: ", epoch)
    print("start learning rate: ", learning_rate)
    print("dimension of latent representation: ", dim)
    print('key: ', key)

    print('whether to use pretrained model: ', use_pretrained)
    print('pretrained_checkpoint: ', pretrained_checkpoint)

    if checkpoint is not None:
        checkpoint = train_dir + checkpoint
        print('pick up checkpoint: ', checkpoint)





    print('load data for Seattle...')
    globals()['TRAINING_STEPS']  = epoch
    globals()['LEARNING_RATE']  = learning_rate
    print('TRAINING_STEPS: ', TRAINING_STEPS)


    intersect_pos = pd.read_csv('../auxillary_data/intersect_pos_32_20.csv')
    intersect_pos_set = set(intersect_pos['0'].tolist())
    # demographic data
    # should use 2018 data
    demo_raw = pd.read_csv('../auxillary_data/whole_grid_32_20_demo_1000_intersect_geodf_2018_corrected.csv', index_col = 0)
    train_obj = train(demo_raw)
    train_obj.generate_binary_demo_attr(intersect_pos_set)

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

    # building_permit_arr_seq = generate_fixlen_timeseries(building_permit_arr, 1)
    # building_permit_arr_seq_extend = np.repeat(building_permit_arr_seq, 24, axis =1)
    # collisions_arr_seq = generate_fixlen_timeseries(collisions_arr, 1)
    # collisions_arr_seq_extend = np.repeat(collisions_arr_seq, 24, axis =1)

    # duplicate building_permit_arr and collisions to the same shape as seattle911calls
    # deal them the same way as 911
    building_permit_arr_seq_extend = np.repeat(building_permit_arr, 24, axis =0)
    collisions_arr_seq_extend = np.repeat(collisions_arr, 24, axis =0)

    print('building_permit_arr_seq_extend.shape: ', building_permit_arr_seq_extend.shape)

    # construct dictionary
    print('use dictionary to organize data')
    # rawdata_1d_dict = {
    #  'weather': weather_arr,
    # # 'airquality': airquality_arr,
    # }
    rawdata_1d_dict_all = {
     'precipitation':  np.expand_dims(weather_arr[:,0], axis=1) ,
    'temperature':  np.expand_dims(weather_arr[:,1], axis=1) ,
    'pressure':  np.expand_dims(weather_arr[:,2], axis=1),
    'airquality': airquality_arr,
    }

    rawdata_2d_dict_all = {
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

    rawdata_3d_dict_all = {
          'building_permit': building_permit_arr_seq_extend,
        'collisions': collisions_arr_seq_extend,  # expect (1, 45984, 32, 20)
        # 'building_permit': building_permit_arr,
        # 'collisions':collisions_arr,
        'seattle911calls': seattle911calls_arr # (45984, 32, 20)
        }

    keys_1d = list(rawdata_1d_dict_all.keys())
    keys_2d = list(rawdata_2d_dict_all.keys())
    keys_3d = list(rawdata_3d_dict_all.keys())


################  read corrputed data ########################

    with open(path_1d + 'rawdata_1d_corrupted_dict', 'rb') as handle:
        rawdata_1d_corrupted_dict_all = pickle.load(handle)

    with open(path_2d + 'rawdata_2d_corrupted_dict', 'rb') as handle:
        rawdata_2d_corrupted_dict_all = pickle.load(handle)

    with open(path_3d + 'rawdata_3d_corrupted_dict', 'rb') as handle:
        rawdata_3d_corrupted_dict_all = pickle.load(handle)


##################  grouping ################################

    grouping_dict = {
    'weather_grp': ['precipitation','temperature', 'pressure', 'airquality'],
    'transportation_grp': ['POI_transportation', 'seattle_street', 'total_flow_count',
                     'transit_routes', 'transit_signals', 'transit_stop', 'bikelane',
                      'collisions', 'slope'],
    'economics_grp': ['house_price', 'POI_business', 'POI_food', 'building_permit',
                             'seattle911calls'],
    'public_service_grp': ['POI_government', 'POI_hospitals', 'POI_publicservices',
                     'POI_recreation', 'POI_school']
                }

    rawdata_1d_dict = {}
    rawdata_2d_dict = {}
    rawdata_3d_dict = {}
    rawdata_1d_corrupted_dict = {}
    rawdata_2d_corrupted_dict = {}
    rawdata_3d_corrupted_dict = {}


    selected_keys = grouping_dict[key]
    for key in selected_keys:
        if key != '' and key in keys_1d:
            temp_var = rawdata_1d_dict_all[key]
            rawdata_1d_dict[key] = temp_var
            temp_var_corrected = rawdata_1d_corrupted_dict_all[key]
            rawdata_1d_corrupted_dict[key] = temp_var_corrected


        if key != '' and key in keys_2d:
            temp_var = rawdata_2d_dict[key]
            rawdata_2d_dict[key] = temp_var
            temp_var_corrected = rawdata_2d_corrupted_dict_all[key]
            rawdata_2d_corrupted_dict[key] = temp_var_corrected

        if key != '' and key in keys_3d:
            temp_var = rawdata_3d_dict[key]
            rawdata_3d_dict[key] = temp_var
            temp_var_corrected = rawdata_3d_corrupted_dict_all[key]
            rawdata_3d_corrupted_dict[key] = temp_var_corrected
    # train_obj.train_hours = datetime_utils.get_total_hour_range(train_obj.train_start_time, train_obj.train_end_time)
    print('train_hours: ', train_obj.train_hours)




####################### city ignorant treatment ################
    # lamda = 0
    # if specified training dir to resume training,
    # the save_path is the same dir as train_dir
    # otherwise, create ta new dir for training
    if suffix == '':
        save_path =  './denoise_groupwise_autoencoder_v0_'+ 'dim'+ str(dim)  +'/'
    else:
        if key == '':
            save_path = './denoise_groupwise_autoencoder_v0_'+ 'dim' + str(dim) +'_'+ suffix  +'/'
        else:
            save_path = './denoise_groupwise_autoencoder_v0_'+ 'dim' + str(dim) + '_'+ suffix+ '_' + key  +'/'

    if train_dir:
        save_path = train_dir

    print("training dir: ", train_dir)
    print("save_path: ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # generate mask arr for city boundary
    demo_mask_arr = train_obj.demo_mask()

    # generate demographic in array format
    print('generating demo_arr array')
    demo_arr = train_obj.selected_demo_to_tensor()
    if not os.path.isfile(save_path +  str(place) + '_demo_arr_' + str(HEIGHT) + '.npy'):
        np.save(save_path + str(place)+ '_demo_arr_'+ str(HEIGHT) + '.npy', demo_arr)


    timer = str(time.time())
    if resume_training == False:
        if inference == False:
        # Model fusion without fairness
            print('Train Model')
            latent_representation = autoencoder_v0.Autoencoder_entry(train_obj,
                                    rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict, intersect_pos_set,
                                     demo_mask_arr,  save_path, dim,
                                HEIGHT, WIDTH, TIMESTEPS, CHANNEL, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                                use_pretrained = use_pretrained, pretrained_ckpt_path = pretrained_checkpoint,
                        ).train_lat_rep
        else:
            latent_representation = autoencoder_v0.Autoencoder_entry(train_obj,
                                        rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict, intersect_pos_set,
                                         demo_mask_arr,  save_path, dim,
                                    HEIGHT, WIDTH, TIMESTEPS, CHANNEL, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                                    True, checkpoint, False, train_dir,
                                    use_pretrained = use_pretrained, pretrained_ckpt_path = pretrained_checkpoint,

                            ).final_lat_rep
    else:
         # resume training
        print('resume trainging from : ', train_dir)
        latent_representation = autoencoder_v0.Autoencoder_entry(train_obj,
                            rawdata_1d_dict, rawdata_2d_dict, rawdata_3d_dict, intersect_pos_set,
                                         demo_mask_arr,
                            train_dir, dim,
                            HEIGHT, WIDTH, TIMESTEPS, CHANNEL,
                            BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                            False, checkpoint, True, train_dir).train_lat_rep
    print('saving latent representation to npy')
    print('shape of latent_representation: ', latent_representation.shape)

    # np.save(save_path +'latent_representation_train.npy', latent_representation)


    txt_name = save_path + 'autoencoder_v2_1to1_' + 'dim_' + str(dim) +'_'  + timer + '.txt'
    with open(txt_name, 'w') as the_file:
        the_file.write('Only account for grids that intersect with city boundary \n')
        the_file.write('place\n')
        the_file.write(str(place) + '\n')
        the_file.write('dim\n')
        the_file.write(str(dim) + '\n')
        the_file.write('learning rate\n')
        the_file.write(str(LEARNING_RATE) + '\n')
        the_file.write('key\n')
        the_file.write(str(key) + '\n')
        the_file.write('selected_keys\n')
        for item in selected_keys:
            the_file.write("%s\n" % item)
        the_file.close()

        the_file.close()



    # calc grad norm

    train_sub_grad_csv_path = save_path + 'autoencoder_train_sub_grad' +'.csv'
    if os.path.exists(train_sub_grad_csv_path):
        test_df = pd.read_csv(train_sub_grad_csv_path, index_col=0)
        test_df = 1/test_df
        test_df = test_df.apply(lambda x: x/x.max(), axis=1)
        test_df.to_csv(save_path + 'autoencoder_v2_grad_normalized' +'.csv')
        print('saved grad norm to : ', save_path + 'autoencoder_v2_grad_normalized' +'.csv')
        last_row_dict = test_df.iloc[-1,:].to_dict()

        last_corr = test_df.iloc[-1,:].corr(test_df.iloc[-2,:])
        print('correlation between last two rows of grad: ', last_corr)

        recon_file = open(save_path + 'grad_dict', 'wb')
        print('saved grad dict to : ',save_path + 'grad_dict')
        pickle.dump(last_row_dict, recon_file)
        recon_file.close()



if __name__ == '__main__':
    main()
