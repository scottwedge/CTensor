#  train autoencoder for urban features
# first, stack features to form a 3D tensor [H, W, time, channels]
# train autoencoder to learn laten representation as [H, W, 1, 1]
# the latent represetnation is learned from a sequece of 168 hours
# and every week is a summary of its 168 hours.

# when used for new task prediction. The latent representation can be
# directly concatenate with 168 hours of historical biekshare data.

# last updated: October, 2019


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
# import convLSTM
# import convLSTM_earlyfusion
# import convLSTM_latefusion
# import conv_3d
# import conv_3d_latefusion
# import conv_3d_baseline
# import conv_3d_mean_diff
# import conv_3d_metric2
# import conv_3d_pairwise
# import fused_model
# import fused_model_augment

import autoencoder_v1
from matplotlib import pyplot as plt
import random



HEIGHT = 32
WIDTH = 20
TIMESTEPS = 168

# without exogenous data, the only channel is the # of trip starts
BIKE_CHANNEL = 1
NUM_2D_FEA = 5 # slope = 2, bikelane = 2, houseprice = 1
NUM_1D_FEA = 3  # temp/slp/prec
CHANNEL = 27  # number of all features

BATCH_SIZE = 32
# actually epochs
# TRAINING_STEPS = 200
TRAINING_STEPS = 50

LEARNING_RATE = 0.001


#fea_list = ['asian_pop','black_pop','hispanic_p', 'no_car_hh','poverty_po',
 #'white_pop','ave_hh_inc','edu_uni','edu_high','hh_incm_hi','hh','pop','age65','poverty_perc']

#fea_list = ['asian_pop','black_pop','hispanic_p', 'no_car_hh','white_pop','edu_uni','edu_high','hh_incm_hi','age65','poverty_perc']
#fea_list = ['pop','normalized_pop', 'bi_caucasian','bi_age', 'bi_high_incm','bi_edu_univ', 'bi_nocar_hh']
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

        self.train_hours = datetime_utils.get_total_hour_range(self.train_start_time, self.train_end_time)

        # self.train_df = raw_df[self.train_start_time: self.train_end_time]
        # self.test_df = raw_df[self.test_start_time: self.test_end_time]
        # self.grid_list = list(raw_df)
        #self.test_df_cut = self.test_df.loc[:,self.test_df.columns.isin(list(self.intersect_pos_set))]


    '''
    Thresholds by defaut are city average in 2018

    - race: non-causasion / causasion: white > 50%
    or caucasion > 65.7384 %
    - age: age 65 > threshold (13.01)
    - income: income above 10k > threshold (41.76)
    - edu (edu_univ: 53.48)
    - no-car (16.94)

    fea_list = ['asian_pop','black_pop','hispanic_p', 'no_car_hh','white_pop',
                'edu_uni','edu_high','hh_incm_hi','age65','poverty_perc']

    # always advantage group : disadvantage group
    # e.g.,: age>65 = 1,  age<65 = -1;   nocar = -1, more car = 1
    '''
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

    '''
    return pop_df for Seattle
                    overall	caucasian	non_caucasian	senior	young	high_incm	low_incm	high_edu	low_edu	fewer_car	more_car
        num_grid	346.0	238.000000	108.000000	221.000000	125.000000	211.000000	135.000000	182.000000	164.000000	50.000000	296.000000
        pop	         1.0	0.695957	0.304043	0.459028	0.540972	0.505706	0.494294	0.564805	0.435195	0.306185	0.693815

    '''
    def generate_pop_df(self):
        pop_cols = ['overall','caucasian', 'non_caucasian', 'senior','young', 'high_incm', 'low_incm',
                     'high_edu','low_edu', 'fewer_car', 'more_car']
        pop_df = pd.DataFrame(0,  index=['num_grid', 'pop'], columns= pop_cols)
        pop_ratio_df = pd.DataFrame(0, index = ['ratio'], columns=['caucasian_non_caucasian',
                                                                                'young_senior', 'high_incm_low_incm',
                                                            'high_edu_low_edu', 'more_car_fewer_car'])
        # iterating through all grids
        for idx, row in self.demo_raw.iterrows():
            grid_num = row['pos']
            if(pd.isnull(row['asian_pop'])):
                continue

            pop_df.loc['num_grid','overall'] += 1
            pop_df.loc['pop','overall'] += row['normalized_pop']

            if row['bi_caucasian'] == 1:
                # gt_equity_df.loc['mean_tripstart','caucasian'] += gt_mean_df[grid_num][0]
                pop_df.loc['num_grid','caucasian'] += 1
                pop_df.loc['pop','caucasian'] += row['normalized_pop']

            if row['bi_caucasian'] == -1:
                    #gt_equity_df.loc['mean_tripstart','non_caucasian'] += gt_mean_df[grid_num][0]
                pop_df.loc['num_grid','non_caucasian'] += 1
                pop_df.loc['pop','non_caucasian'] += row['normalized_pop']

            if row['bi_age'] == 1:
                    #gt_equity_df.loc['mean_tripstart','senior'] += gt_mean_df[grid_num][0]
                pop_df.loc['num_grid','young'] += 1
                pop_df.loc['pop','young'] += row['normalized_pop']

            if row['bi_age'] == -1:
                    #gt_equity_df.loc['mean_tripstart','young'] += gt_mean_df[grid_num][0]
                pop_df.loc['num_grid','senior'] += 1
                pop_df.loc['pop','senior'] += row['normalized_pop']


            if row['bi_high_incm'] == 1:
                    #gt_equity_df.loc['mean_tripstart','high_incm'] += gt_mean_df[grid_num][0]
                pop_df.loc['num_grid','high_incm'] += 1
                pop_df.loc['pop','high_incm'] += row['normalized_pop']

            if row['bi_high_incm'] == -1:
                    #gt_equity_df.loc['mean_tripstart','low_incm'] += gt_mean_df[grid_num][0]
                pop_df.loc['num_grid','low_incm'] += 1
                pop_df.loc['pop','low_incm'] += row['normalized_pop']

            if row['bi_edu_univ'] == 1:
                    #gt_equity_df.loc['mean_tripstart','high_edu'] += gt_mean_df[grid_num][0]
                pop_df.loc['num_grid','high_edu'] += 1
                pop_df.loc['pop','high_edu'] += row['normalized_pop']
            if row['bi_edu_univ'] == -1:
                    #gt_equity_df.loc['mean_tripstart','low_edu'] += gt_mean_df[grid_num][0]
                pop_df.loc['num_grid','low_edu'] += 1
                pop_df.loc['pop','low_edu'] += row['normalized_pop']

            if row['bi_nocar_hh'] == 1:
                    #gt_equity_df.loc['mean_tripstart','fewer_car'] += gt_mean_df[grid_num][0]
                pop_df.loc['num_grid','more_car'] += 1
                pop_df.loc['pop','more_car'] += row['normalized_pop']
            if row['bi_nocar_hh'] == -1:
                    #gt_equity_df.loc['mean_tripstart','more_car'] += gt_mean_df[grid_num][0]
                pop_df.loc['num_grid','fewer_car'] += 1
                pop_df.loc['pop','fewer_car'] += row['normalized_pop']


            pop_ratio_df.loc['ratio','caucasian_non_caucasian'] = pop_df['caucasian']['pop'] / pop_df['non_caucasian']['pop']
            pop_ratio_df.loc['ratio','young_senior'] = pop_df['young']['pop'] /pop_df['senior']['pop']

            pop_ratio_df.loc['ratio','high_incm_low_incm'] = pop_df['high_incm']['pop'] /pop_df['low_incm']['pop']

            pop_ratio_df.loc['ratio','high_edu_low_edu'] = pop_df['high_edu']['pop'] /pop_df['low_edu']['pop']
            pop_ratio_df.loc['ratio','more_car_fewer_car'] = pop_df['more_car']['pop'] /pop_df['fewer_car']['pop']

        return pop_df,pop_ratio_df


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


    # for each image randomly sample pixels
    def add_noise_to_one_image(self,img):
        height = img.shape[1] # 20
        width = img.shape[0] # 32
        # num of pixels to add noise, range from 5% to 10 % of total pixel num (32 - 64 pixels)
        num_pixel = random.choice(range(int(height * width * 0.05)+1,  int(height * width * 0.1)+1))
        # generate a sequence of random numbers from 0 to 640.
        pixel_to_use = set(random.sample(range(0, height * width), num_pixel))
        index = 0
        # apply gaussian noise to sampled pixel,
        for p in pixel_to_use:
            c = int(p / width)-1  # 0 -20  -> height
            r = int(p % width)-1  # 0 -32
            # generate a random noise
            noise = int(random.gauss(0, 2))
            img[r, c] = noise + img[r, c]
            if img[r, c]< 0:
                img[r, c] = 0
        return img


    # rawdata shape: (9504, 32, 20)
    # note: make changes in place
    def generate_noise_data(self, rawdata_arr):
        noisy_data_list = []
        for i in range(rawdata_arr.shape[0]):
            #print('adding noise to ith data: ', i)
            noisy_data_list.append(self.add_noise_to_one_image(rawdata_arr[i,:,:]))
        noisy_data_arr = np.array(noisy_data_list)
        return noisy_data_arr





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',   '--suffix',
                     action="store", help = 'save path suffix', default = '')
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
                     action="store", help = 'epochs to train', default = 0.001)


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

    print("resume_training: ", resume_training)
    print("training dir path: ", train_dir)
    print("checkpoint: ", checkpoint)
    print("place: ", place)
    print("epochs to train: ", epoch)
    print("start learning rate: ", learning_rate)

    if checkpoint is not None:
        checkpoint = train_dir + checkpoint
        print('pick up checkpoint: ', checkpoint)


    if place == "Seattle":
        print('load data for Seattle...')
        globals()['TRAINING_STEPS']  = epoch
        globals()['LEARNING_RATE']  = learning_rate
        print('TRAINING_STEPS: ', TRAINING_STEPS)

        # hourly_grid_timeseries = pd.read_csv('./hourly_grid_1000_timeseries_trail.csv', index_col = 0)
        # hourly_grid_timeseries.index = pd.to_datetime(hourly_grid_timeseries.index)
        # rawdata = pd.read_csv('lime_whole_grid_32_20_hourly_1000_171001-181031.csv', index_col = 0)
        # rawdata.index = pd.to_datetime(rawdata.index)
        # a set of region codes (e.g.: 10_10) that intersect with the city
        intersect_pos = pd.read_csv('../auxillary_data/intersect_pos_32_20.csv')
        intersect_pos_set = set(intersect_pos['0'].tolist())
        # demographic data
        # should use 2018 data
        demo_raw = pd.read_csv('../auxillary_data/whole_grid_32_20_demo_1000_intersect_geodf_2018_corrected.csv', index_col = 0)
        train_obj = train(demo_raw)
        #ignore non-intersection cells in test_df
        # this is for evaluation
        # test_df_cut = train_obj.test_df.loc[:,train_obj.test_df.columns.isin(list(intersect_pos_set))]
        # generate binary demo feature according to 2018 city mean
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

        print('stack 1d data')
        weather_arr = weather_arr[0,0,:,:]
        airquality_arr = airquality_arr[0,0,:,:]
        datalist_1d = [weather_arr, airquality_arr]
        data_1d = np.concatenate(datalist_1d, axis=1)
        print('data_1d.shape: ', data_1d.shape)

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

        print('Stack 2d data')
                # stack 2d data
        datalist_2d = [house_price_arr,POI_business_arr, POI_food_arr, POI_government_arr,
                              POI_hospitals_arr, POI_publicservices_arr, POI_recreation_arr, POI_school_arr,
                              POI_transportation_arr, seattle_street_arr, total_flow_count_arr, transit_routes_arr,
                              transit_signals_arr, transit_stop_arr, slope_arr, bikelane_arr]
        data_2d = np.concatenate(datalist_2d, axis=2)
        print('data_2d.shape: ', data_2d.shape)


        # 3d
        building_permit_arr = np.load(path_3d + 'building_permit_arr_20140201_20190501_python3.npy')
        collisions_arr = np.load(path_3d + 'collisions_arr_20140201_20190501_python3.npy',encoding='latin1', allow_pickle=True)
        crime_arr = np.load(path_3d + 'crime_arr_20140201_20190501_python3.npy')
        seattle911calls_arr = np.load(path_3d + 'seattle911calls_arr_20140201_20190501.npy')
        print('building_permit_arr.shape:', building_permit_arr.shape)
        print('collisions_arr.shape: ', collisions_arr.shape)
        print('crime_arr.shape: ', crime_arr.shape)
        print('seattle911calls_arr.shape: ', seattle911calls_arr.shape)

        print('stack 3d')
        building_permit_extend_arr = np.repeat(building_permit_arr, 24, axis =0)
        collisions_extend_arr = np.repeat(collisions_arr, 24, axis =0)
        building_permit_extend_arr = np.expand_dims(building_permit_extend_arr, axis=3)
        collisions_extend_arr = np.expand_dims(collisions_extend_arr, axis=3)
        seattle911calls_arr = np.expand_dims(seattle911calls_arr, axis=3)
        datalist_3d = [seattle911calls_arr, building_permit_extend_arr, collisions_extend_arr]
        data_3d = np.concatenate(datalist_3d, axis=3)
        print('data_3d.shape: ', data_3d.shape)

        # train_obj.train_hours = datetime_utils.get_total_hour_range(train_obj.train_start_time, train_obj.train_end_time)
        print('train_hours: ', train_obj.train_hours)

        # -----  load bike data ------ #
        # if os.path.isfile('bikedata_32_20_171001-181031.npy'):
        #     print('loading raw data array...')
        #     rawdata_arr = np.load('bikedata_32_20_171001-181031.npy')
        # else:
        #     print('generating raw data array')
        #     rawdata_arr = train_obj.df_to_tensor()
        #     np.save('bikedata_32_20_171001-181031.npy', rawdata_arr)

####################### city ignorant treatment ################
    # lamda = 0
    # if specified training dir to resume training,
    # the save_path is the same dir as train_dir
    # otherwise, create ta new dir for training
    if suffix == '':
        save_path =  './autoencoder_v1_'+ str(place) +'/'
    else:
        save_path = './autoencoder_v1_'+ str(place) + suffix  +'/'

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
    # Model fusion without fairness
        print('Train Model')
        latent_representation = autoencoder_v1.Autoencoder_entry(train_obj, data_1d, data_2d, data_3d, intersect_pos_set,
                                 demo_mask_arr,  save_path,
                            HEIGHT, WIDTH, TIMESTEPS, CHANNEL, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE
                    ).latent_representation
    else:
         # resume training
        print('resume trainging from : ', train_dir)
        latent_representation = autoencoder_v1.Autoencoder_entry(train_obj, train_arr, test_arr, intersect_pos_set,
                                             weather_seq_arr, crime_seq_arr, data_2d,
                                            # multi_demo_sensitive, demo_pop, multi_pop_g1, multi_pop_g2,
                                            # multi_grid_g1, multi_grid_g2,fairloss,
                                            # train_arr_1d, test_arr_1d, data_2d,
                                        lamda, demo_mask_arr,
                            train_dir, beta,
                            HEIGHT, WIDTH, TIMESTEPS, CHANNEL,
                     NUM_2D_FEA, NUM_1D_FEA, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                            False, checkpoint, True, train_dir).latent_representation
    print('saving latent representation to npy')

    np.save(save_path +'latent_representation.npy', latent_representation)


    txt_name = save_path + 'autoencoder_v1_' +  timer + '.txt'
    with open(txt_name, 'w') as the_file:
        the_file.write('Only account for grids that intersect with city boundary \n')
        the_file.write('place\n')
        the_file.write(str(place) + '\n')
        the_file.write('learning rate\n')
        the_file.write(str(LEARNING_RATE) + '\n')


        the_file.close()



if __name__ == '__main__':
    main()
