
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
import convLSTM
import convLSTM_earlyfusion
import convLSTM_latefusion
import conv_3d
import conv_3d_latefusion
import conv_3d_baseline
import conv_3d_mean_diff
import conv_3d_metric2
import conv_3d_pairwise
import fused_model
import fused_model_augment

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
CHANNEL = 9  # number of all features 


BATCH_SIZE = 32
# actually epochs
TRAINING_STEPS = 200
#TRAINING_STEPS = 10

LEARNING_RATE = 0.001


# without exogenous data, the only channel is the # of trip starts
# CHANNEL = 1
# BATCH_SIZE = 32
# TRAINING_STEPS = 1000

#fea_list = ['asian_pop','black_pop','hispanic_p', 'no_car_hh','poverty_po',
 #'white_pop','ave_hh_inc','edu_uni','edu_high','hh_incm_hi','hh','pop','age65','poverty_perc']

#fea_list = ['asian_pop','black_pop','hispanic_p', 'no_car_hh','white_pop','edu_uni','edu_high','hh_incm_hi','age65','poverty_perc']
#fea_list = ['pop','normalized_pop', 'bi_caucasian','bi_age', 'bi_high_incm','bi_edu_univ', 'bi_nocar_hh']
fea_list = ['pop','normalized_pop', 'bi_caucasian','bi_age', 'bi_high_incm','bi_edu_univ', 'bi_nocar_hh', 
           'white_pop','age65_under', 'edu_uni']



class train:
    # TODO: increase window size to 4 weeks
    def __init__(self, raw_df, demo_raw, 
                train_start_time = '2017-10-01',train_end_time = '2018-08-31',
                test_start_time = '2018-09-01 00:00:00', test_end_time = '2018-10-31 23:00:00' ):
        self.raw_df = raw_df
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

        self.train_df = raw_df[self.train_start_time: self.train_end_time]
        self.test_df = raw_df[self.test_start_time: self.test_end_time]
        self.grid_list = list(raw_df)

        # 
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
    parser.add_argument('lamda', nargs='?', type = float, help = 'lambda for fairness', default = 0)
    parser.add_argument('beta', nargs='?', type = float, help = 'beta for weighted MAE; gamma for differential weighted MAE;and for binary weight', default = 1.0)
    # parser.add_argument('-use_1d_fea',   type=bool, default=False,
    #                 action="store", help = 'whether to use 1d features. If use this option, set to True. Otherwise, default False')
    # parser.add_argument('-use_2d_fea',    type=bool, default=False,
    #                 action="store", help = 'whether to use 2d features')
    # parser.add_argument('-fairloss',    type=str, default='IFG',
    #                 action="store", help = 'whether to fairloss: IFG, RFG, equalmean, pairwise')
    # parser.add_argument('-multivar',    type=bool, default=False,
    #                 action="store", help = 'whether to multi-var fairloss. If True, include aga, race, and edu. Otherwise, use race')
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
                     action="store", help = 'epochs to train', default = 200)
    parser.add_argument('-l',   '--learning_rate',  type=float,
                     action="store", help = 'epochs to train', default = 0.001)
   
    
    #parser.add_argument('-a','--use_1d_fea', type=bool, default=True,  
     #                action="store", help = 'whether to use 1d features')
    #parser.add_argument('-b','--use_2d_fea', type=bool, default=True,
     #                 action="store", help = 'whether to use 2d features')
    # parser.add_argument('city_stat', help = 'city-wide demographic data in csv format')
    return parser.parse_args()



def main():
    args = parse_args()
    lamda = args.lamda
    beta = args.beta
    # use_1d_fea = bool(args.use_1d_fea)
    # use_2d_fea = bool(args.use_2d_fea)
    # fairloss = args.fairloss
    # multivar=  bool(args.multivar)
    suffix = args.suffix

    # the following arguments for resuming training
    resume_training = args.resume_training
    train_dir = args.train_dir
    checkpoint = args.checkpoint
    place = args.place
    epoch = args.epoch
    learning_rate= args.learning_rate
    
    print("received arguments: lamda: ",lamda)
    print("received arguments: beta: ",beta)

    # print("use_1d_fea: ", use_1d_fea)
    # print("use_2d_fea: ", use_2d_fea)
    # print("fairloss: ", fairloss)
    # print("multivar: ", multivar)
    print("resume_training: ", resume_training)
    print("training dir path: ", train_dir)
    print("checkpoint: ", checkpoint)
    print("place: ", place)
    print("epochs to train: ", epoch)
    print("start learning rate: ", learning_rate)

    # if checkpoint is not None:
    #     checkpoint = train_dir + checkpoint
    #     print('pick up checkpoint: ', checkpoint)


        # get checkpoint 
    # if provided, use the checkpoint, otherwise, get most recent checkpoint
    # could also use: https://www.tensorflow.org/api_docs/python/tf/train/latest_checkpoint
    # e.g.: saver.restore(sess, tf.train.latest_checkpoint('./')) 

    if checkpoint is not None:
        checkpoint = train_dir + checkpoint
    else:
        # get most recently checkpoint
        # see https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/checkpoint_management.py
        # get filenames containing ckpt
        all_cp = set([d for d in os.listdir(train_dir) if 'ckpt' in d])
        cp_num = 0  # 207
        prefix = ''  # fusion_model_0.0_207 
        ckpt_midfix = ''  # ckpt-1006
        # fusion_model_0.0_290.ckpt.index  or fusion_model_0.0_1.ckpt-1006.index
        # 1006 = global step
        for cp in all_cp:
            # fusion_model_0.0_207.ckpt.index
            temp_prefix_list = cp.split('.')[0:-2]
            temp_prefix = '.'.join(temp_prefix_list)
            temp_num = int(temp_prefix.split('_')[-1])
            if temp_num > cp_num:
                cp_num = temp_num
                prefix = temp_prefix
                ckpt_midfix = cp.split('.')[-2]
           
        # latest checkpoint name

        checkpoint = prefix + '.'+ckpt_midfix
        checkpoint = train_dir + checkpoint
    print('pick up checkpoint: ', checkpoint)


    if place == "Seattle":
        print('load data for Seattle...')
        globals()['TRAINING_STEPS']  = epoch
        globals()['LEARNING_RATE']  = learning_rate
        print('TRAINING_STEPS: ', TRAINING_STEPS)

        # hourly_grid_timeseries = pd.read_csv('./hourly_grid_1000_timeseries_trail.csv', index_col = 0)
        # hourly_grid_timeseries.index = pd.to_datetime(hourly_grid_timeseries.index)
        rawdata = pd.read_csv('lime_whole_grid_32_20_hourly_1000_171001-181031.csv', index_col = 0)
        rawdata.index = pd.to_datetime(rawdata.index)
        # a set of region codes (e.g.: 10_10) that intersect with the city
        intersect_pos = pd.read_csv('intersect_pos_32_20.csv')
        intersect_pos_set = set(intersect_pos['0'].tolist())
        # demographic data
        # should use 2018 data
        demo_raw = pd.read_csv('../UW_lib/whole_grid_32_20_demo_1000_intersect_geodf_2018_corrected.csv', index_col = 0)
        train_obj = train(rawdata, demo_raw)
        #ignore non-intersection cells in test_df
        # this is for evaluation
        # test_df_cut = train_obj.test_df.loc[:,train_obj.test_df.columns.isin(list(intersect_pos_set))]
        # generate binary demo feature according to 2018 city mean
        train_obj.generate_binary_demo_attr(intersect_pos_set)

      
        bikelane_arr = np.load('../feature_transform/bikelane_arr.npy')
        slope_arr = np.load('../feature_transform/slope_arr.npy')
        house_price_arr = np.load('../feature_transform/seattle_houseprice_arr.npy')
        data_2d = np.concatenate([slope_arr,bikelane_arr], axis=2)
        data_2d = np.concatenate([data_2d,house_price_arr], axis=2)
            # transitstop_arr = np.load('../feature_transform/transitstop_arr.npy')
            # weather: (1,1,9504,3) or (9504, 3)
        weather_arr = np.load('../feature_transform/weather_arr_1by1by9504.npy')
        print('weather_arr.shape: ', weather_arr.shape)
        # weather_arr = weather_arr[0,0,:,:]  # [9504, 3]
            # construct training / testing data for 1d data
        print('generating fixed window length training and testing sequences for 1d data')
        weather_seq_arr = train_obj.generate_fixlen_timeseries(weather_arr)
       
            # test_series_1d.shape -> (169, 1296, 3)
            # train_arr_1d, test_arr_1d = train_obj.train_test_split(raw_seq_arr_1d)
        # load crime data
        crime_arr = np.load('../st_data/crimearr_32_20_171001-181031.npy')
        crime_seq_arr = train_obj.generate_fixlen_timeseries(crime_arr)

        # swapping axis
        crime_seq_arr = np.swapaxes(crime_seq_arr,0,1)
        weather_seq_arr = np.swapaxes(weather_seq_arr,0,1)

        print('crime_seq_arr.shape: ', crime_seq_arr.shape)
        print('weather_seq_arr.shape: ', weather_seq_arr.shape)
       
        if os.path.isfile('bikedata_32_20_171001-181031.npy'):  
            print('loading raw data array...')
            rawdata_arr = np.load('bikedata_32_20_171001-181031.npy')
        else:
            print('generating raw data array')
            rawdata_arr = train_obj.df_to_tensor()
            np.save('bikedata_32_20_171001-181031.npy', rawdata_arr)

    elif place == "Austin":
        print('load data for Austin...')
        globals()['HEIGHT']  = 28
        globals()['WIDTH']  = 28
        globals()['TIMESTEPS']  = 168
        globals()['BIKE_CHANNEL']  = 1
        globals()['NUM_2D_FEA']  = 3  # street count / streent len / poi count
        globals()['NUM_1D_FEA']  = 3
        globals()['BATCH_SIZE']  = 32
        globals()['TRAINING_STEPS']  = epoch
        # globals()['LEARNING_RATE']  = 0.003
        globals()['LEARNING_RATE']  = learning_rate
        print('global HEIGHT: ', HEIGHT)

        train_start_time = '2016-08-01'
        train_end_time = '2017-02-28'
        test_start_time = '2017-03-01 00:00:00'
        test_end_time = '2017-04-13 23:00:00' 
        print('train_start_time for Austin: ', train_start_time)

        # hourly_grid_timeseries = pd.read_csv('./hourly_grid_1000_timeseries_trail.csv', index_col = 0)
        # hourly_grid_timeseries.index = pd.to_datetime(hourly_grid_timeseries.index)
        rawdata = pd.read_csv('../rideaustin/rideaustin_grided_hourly_2000_20160801-20170413.csv', index_col = 0)
        rawdata.index = pd.to_datetime(rawdata.index)

        # a set of region codes (e.g.: 10_10) that intersect with the city
        intersect_pos = pd.read_csv('../rideaustin/austin_intersect_pos_28_28.csv')
        intersect_pos_set = set(intersect_pos['0'].tolist())
        # demographic data
        # should use 2018 data
        demo_raw = pd.read_csv('../rideaustin/austin_demo_data/austin_28_28_demo_2000_intersect_geodf_2017.csv', index_col = 0)
        train_obj = train(rawdata, demo_raw, 
                train_start_time, train_end_time, test_start_time, test_end_time)
        #ignore non-intersection cells in test_df
        # this is for evaluation
        test_df_cut = train_obj.test_df.loc[:,train_obj.test_df.columns.isin(list(intersect_pos_set))]
       
        # generate binary demo feature according to 2017 Austin city mean
        train_obj.generate_binary_demo_attr(intersect_pos_set, 70.2222, 8.7057,
           32.6351, 42.0087, 6.453)

        '''
        # load 2d and 1d features
        if use_2d_fea:
            print("use 2d feature")
            # landuse arr 28 28 1
            landuse_arr = np.load('../feature_transform/austin_landuse_arr.npy')
            street_arr = np.load('../feature_transform/austin_street_arr.npy')
            # concatenate 2d data
            data_2d = np.concatenate([landuse_arr,street_arr], axis=2)
        else:
            print('ignore 2d data')
            data_2d = None
        

        if use_1d_fea:
            # weather: (1,1,6144,3)
            weather_arr = np.load('../feature_transform/austin_weather_arr_1by1bytime.npy')
            weather_arr = weather_arr[0,0,:,:]  # [6144, 3]
            # construct training / testing data for 1d data
            print('generating fixed window length training and testing sequences for 1d data')
            raw_seq_arr_1d = train_obj.generate_fixlen_timeseries(weather_arr)
            # test_series_1d.shape -> (169, 1296, 3)
            train_arr_1d, test_arr_1d = train_obj.train_test_split(raw_seq_arr_1d)
            # 
        else:
            print('ignore 1d data')
            train_arr_1d = None
            test_arr_1d = None
        '''
        
        if os.path.isfile('../rideaustin/austin_28_20160801-20170413.npy'):  
            print('loading raw data array...')
            rawdata_arr = np.load('../rideaustin/austin_28_20160801-20170413.npy')
        else:
            print('generating raw data array')
            rawdata_arr = train_obj.df_to_tensor()
            np.save('../rideaustin/austin_28_20160801-20170413.npy', rawdata_arr)
    else:
        print("Please input correct city name")
         



####################### city ignorant treatment ################
    # lamda = 0
    # if specified training dir to resume training, 
    # the save_path is the same dir as train_dir
    # otherwise, create ta new dir for training
    # if suffix == '':
    #     save_path =  './autoencoder_'+ str(place) + '_'+ str(lamda)+'_'+  str(beta) + '/'
    # else:
    #     save_path = './autoencoder_'+ str(place) + '_'+'_' + str(lamda)+'_'+  str(beta)+'_'+ suffix  +'/'

    # if train_dir:
    #     save_path = train_dir

    # print("training dir: ", train_dir)
    # print("save_path: ", save_path)            
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    save_path = train_dir + 'inference/'
    print('inference dir: ', save_path)
    if not os.path.exists(save_path):



    # save demongraphic array
    #if os.path.isfile(save_path + 'demo_arr_32_20.npy'):
     #   print('loading demopgraphic data array...')
      #  demo_arr = np.load(save_path + 'demo_arr_32_20.npy')
    #else:

    # generate mask arr for city boundary
    demo_mask_arr = train_obj.demo_mask()

    # generate demographic in array format
    print('generating demo_arr array')
    demo_arr = train_obj.selected_demo_to_tensor()
    if not os.path.isfile(save_path +  str(place) + '_demo_arr_' + str(HEIGHT) + '.npy'):
        np.save(save_path + str(place)+ '_demo_arr_'+ str(HEIGHT) + '.npy', demo_arr)

    print('generating fixed window length training and testing sequences...')
    raw_seq_arr = train_obj.generate_fixlen_timeseries(rawdata_arr)
    train_arr, test_arr = train_obj.train_test_split(raw_seq_arr)
    print('input train_arr shape: ',train_arr.shape )


    '''
    # add noise and permutation for bike data
    noisy_data_arr = rawdata_arr.copy()
    noisy_data_arr = train_obj.generate_noise_data(noisy_data_arr)

    noisy_seq_arr = train_obj.generate_fixlen_timeseries(noisy_data_arr)
    noisy_train_arr,_ = train_obj.train_test_split(noisy_seq_arr)

    # stack original training data with noisy data
    # shape: (169, 16080, 32, 20)
    comb_train_series =  np.concatenate([train_arr,noisy_train_arr], axis=1)


    
    # permutation
    # perm = np.random.permutation(comb_train_series.shape[1])
    # comb_train_series = comb_train_series[:, perm]

    # permutation for 1d data
    if use_1d_fea:
        copy_train_series_1d = train_arr_1d.copy()
        comb_train_series_1d = np.concatenate([train_arr_1d,copy_train_series_1d], axis=1)
        # perturb as 3d data
        # comb_train_series_1d = comb_train_series_1d[:, perm]
    else:
        comb_train_series_1d = None
    '''

    # calculate statistics for demo
    pop_df, pop_ratio_df = train_obj.generate_pop_df()
    pop_df.to_csv(save_path + 'pop_df.csv')
    pop_ratio_df.to_csv(save_path + 'pop_ratio_df.csv')

    

    # demo_pop: if IFG, RFG, equal mean, use normalized pop. 
    # if pairwise, use non-normalized pop
    # if fairloss == "pairwise":
    # #demo_pop = demo_arr[:,:,1]  # normalized pop
    #     demo_pop = demo_arr[:,:,0]  #  pop # use pop for pairwise loss
    # else:
    #     demo_pop = demo_arr[:,:,1]  # normalized pop
    # demo_pop = np.expand_dims(demo_pop, axis=2)
    # print('demo_pop.shape: ',  demo_pop.shape)

    # demo sensitive 
    '''
    ['pop','normalized_pop','bi_caucasian','bi_age','bi_high_incm',
    'bi_edu_univ','bi_nocar_hh','white_pop','age65_under','edu_uni']
    '''
    # demo_sensitive = demo_arr[:,:,2]  # caucasian
    # demo_sensitive = np.expand_dims(demo_sensitive, axis=2)

    # normalized population of each group
    '''
    caucasian	non_caucasian	senior	young	high_incm	low_incm	
    high_edu	low_edu	  fewer_car	more_car
    '''

    '''
    pop_g1 = pop_df['caucasian'].values[1]
    pop_g2 = pop_df['non_caucasian'].values[1]

    if fairloss == 'RFG':  # metric1: region-based 
        if multivar:
            print('MULTIVAR')
            fea_dim = [2,3,5]  # caucasian, age, edu_univ
            multi_pop_g1 = [pop_df['caucasian'].values[1], pop_df['young'].values[1], pop_df['high_edu'].values[1]]
            multi_pop_g2 = [pop_df['non_caucasian'].values[1], pop_df['senior'].values[1], pop_df['low_edu'].values[1]]
        else:  # single var
            fea_dim = [2]  # binary caucasian 
            # multi_demo_sensitive = demo_arr[:,:,fea_dim]  # caucasian
            multi_pop_g1 = [pop_df['caucasian'].values[1]]
            multi_pop_g2 = [pop_df['non_caucasian'].values[1]]
    elif fairloss == "IFG":
        if multivar:
            print('MULTIVAR')
            fea_dim = [7, 8, 9]  # multivar
        else:
            fea_dim = [7]  # white percent 
        # multi_demo_sensitive = demo_arr[:,:,fea_dim]  # caucasian
        multi_pop_g1 = [pop_df['caucasian'].values[1], pop_df['young'].values[1], pop_df['high_edu'].values[1]]
        multi_pop_g2 = [pop_df['non_caucasian'].values[1], pop_df['senior'].values[1], pop_df['low_edu'].values[1]]
    elif fairloss == "equalmean":
        fea_dim = [2]  # binar caucasian 
        # multi_demo_sensitive = demo_arr[:,:,fea_dim]  # caucasian
        # multi_pop_g1 = [pop_df['caucasian'].values[1], pop_df['young'].values[1], pop_df['high_edu'].values[1]]
        # multi_pop_g2 = [pop_df['non_caucasian'].values[1], pop_df['senior'].values[1], pop_df['low_edu'].values[1]]
        multi_pop_g1 = [pop_df['caucasian'].values[1]]
        multi_pop_g2 = [pop_df['non_caucasian'].values[1]]

        # multi_grid_g1 = [pop_df['caucasian'].values[0]]
        # multi_grid_g2 = [pop_df['non_caucasian'].values[0]]
    elif fairloss == "pairwise":
        multi_pop_g1 = [pop_df['caucasian'].values[1]]
        multi_pop_g2 = [pop_df['non_caucasian'].values[1]]
        fea_dim = [2]  # binar caucasian 
        
    multi_demo_sensitive = demo_arr[:,:,fea_dim]  # caucasian
    multi_grid_g1 = [pop_df['caucasian'].values[0]]  # only for equal mean 
    multi_grid_g2 = [pop_df['non_caucasian'].values[0]]


    # multi-var fairness input
    #fea_dim = [2,3,5]  # caucasian, age, edu_univ
    # fea_dim = [7]  # white percent 
    # multi_demo_sensitive = demo_arr[:,:,fea_dim]  # caucasian

    # multi_pop_g1 = [pop_df['caucasian'].values[1], pop_df['young'].values[1], pop_df['high_edu'].values[1]]
    # multi_pop_g2 = [pop_df['non_caucasian'].values[1], pop_df['senior'].values[1], pop_df['low_edu'].values[1]]
    '''

    timer = str(time.time())
    # if resume_training == False:
    # Model fusion without fairness
    print('Test Model fusion without fairness')
    latent_representation = autoencoder_v1.Autoencoder_entry(train_obj, train_arr, test_arr, intersect_pos_set,
                                            weather_seq_arr, crime_seq_arr, data_2d,
                                            # multi_demo_sensitive, demo_pop, multi_pop_g1, multi_pop_g2,
                                            # multi_grid_g1, multi_grid_g2, fairloss, 
                                            # train_arr_1d, test_arr_1d, data_2d,
                                        lamda, demo_mask_arr,
                            save_path, beta,
                            HEIGHT, WIDTH, TIMESTEPS, CHANNEL, 
                     NUM_2D_FEA, NUM_1D_FEA, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
                        True, checkpoint).latent_representation
    # else:
    #      # resume training
    #     print('resume trainging from : ', train_dir)
    #     latent_representation = autoencoder_v1.Autoencoder_entry(train_obj, train_arr, test_arr, intersect_pos_set,
    #                                          weather_seq_arr, crime_seq_arr, data_2d,
    #                                         # multi_demo_sensitive, demo_pop, multi_pop_g1, multi_pop_g2,
    #                                         # multi_grid_g1, multi_grid_g2,fairloss,
    #                                         # train_arr_1d, test_arr_1d, data_2d,
    #                                     lamda, demo_mask_arr,
    #                         train_dir, beta, 
    #                         HEIGHT, WIDTH, TIMESTEPS, CHANNEL, 
    #                  NUM_2D_FEA, NUM_1D_FEA, BATCH_SIZE, TRAINING_STEPS, LEARNING_RATE,
    #                         False, checkpoint, True, train_dir).latent_representation
    print('saving latent representation to npy')

    np.save(save_path +'infer_latent_representation.npy', latent_representation)
    

    # conv3d_predicted.index = pd.to_datetime(conv3d_predicted.index)
    # conv3d_predicted.to_csv(save_path + 'fused_model_pred_'+ timer + '.csv')
        #convlstm_predicted = pd.read_csv(save_path + 'convlstm_predicted.csv', index_col=0)
        #convlstm_predicted.index = pd.to_datetime(convlstm_predicted.index)
    # eval_obj4 = evaluation.evaluation(test_df_cut, conv3d_predicted, train_obj.demo_raw)
    # diff_df = eval_obj4.group_difference()
    # diff_df.to_csv(save_path+ str(place) +'_evaluation.csv')

    # finegrain_diff_df = eval_obj4.individual_difference()
    # finegrain_diff_df.to_csv(save_path+'IFG_eval.csv')

    # print('rmse for conv3d: ',eval_obj4.rmse_val)
    # print('mae for conv3d: ', eval_obj4.mae_val)

    # plot train test accuracy
    # train_test = pd.read_csv(save_path  + 'ecoch_res_df_' + str(lamda)+'.csv')
    # train_test = train_test.loc[:, ~train_test.columns.str.contains('^Unnamed')]
    # total_loss = train_test[['train_loss', 'test_loss']].plot()
    # plt.savefig(save_path + 'total_loss_finish.png')
    # acc_loss = train_test[['train_acc', 'test_acc']].plot()
    # plt.savefig(save_path + 'acc_loss_finish.png')
    # fair_loss = train_test[['train_fair', 'test_fair']].plot()
    # plt.savefig(save_path + 'fair_loss_finish.png')
    # plt.close()



    txt_name = save_path + 'autoencoder_' +str(lamda)+'_'+   str(beta)+'_'+   timer + '.txt'
    with open(txt_name, 'w') as the_file:
        the_file.write('Only account for grids that intersect with city boundary \n')
        the_file.write('lamda\n')
        the_file.write(str(lamda) + '\n')
        the_file.write('beta\n')
        the_file.write(str(beta) + '\n')
        the_file.write('place\n')
        the_file.write(str(place) + '\n')
        # the_file.write('use_1d_fea\n')
        # the_file.write(str(use_1d_fea) + '\n')
        # the_file.write('use_2d_fea\n')
        # the_file.write(str(use_2d_fea) + '\n')
        # the_file.write('fairloss\n')
        # the_file.write(str(fairloss) + '\n')
        # the_file.write('multivar\n')
        # the_file.write(str(multivar) + '\n')
        the_file.write('learning rate\n')
        the_file.write(str(LEARNING_RATE) + '\n')

        # the_file.write('rmse for conv3d\n')
        # the_file.write(str(eval_obj4.rmse_val) + '\n')
        # the_file.write('mae for conv3d\n')
        # the_file.write(str(eval_obj4.mae_val)+ '\n')

        # the_file.write('mean_diff_percap for bi_caucasian: \n')
        # the_file.write(str(diff_df['bi_caucasian']['mean_diff_percap'])+ '\n')

        # the_file.write('mean_diff_percap for bi_age: \n')
        # the_file.write(str(diff_df['bi_age']['mean_diff_percap'])+ '\n')

        # the_file.write('mean_diff_percap for bi_high_incm: \n')
        # the_file.write(str(diff_df['bi_high_incm']['mean_diff_percap'])+ '\n')

        # the_file.write('mean_diff_percap for bi_edu_univ: \n')
        # the_file.write(str(diff_df['bi_edu_univ']['mean_diff_percap'])+ '\n')

        # the_file.write('mean_diff_percap for bi_nocar_hh: \n')
        # the_file.write(str(diff_df['bi_nocar_hh']['mean_diff_percap'])+ '\n')

        # the_file.write('individual difference for caucasian_non_caucasian: \n')
        # the_file.write(str(finegrain_diff_df['caucasian_non_caucasian']['diff'])+ '\n')

        # the_file.write('individual difference for young_senior: \n')
        # the_file.write(str(finegrain_diff_df['young_senior']['diff'])+ '\n')

        # the_file.write('individual difference for high_edu_low_edu: \n')
        # the_file.write(str(finegrain_diff_df['high_edu_low_edu']['diff'])+ '\n')


        the_file.close()



if __name__ == '__main__':
    main()
    

