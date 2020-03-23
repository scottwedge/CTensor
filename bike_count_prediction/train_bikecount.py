# train LSTM for Fremont bridge bike count
# https://data.seattle.gov/Transportation/Fremont-SB-bicycle-count/aggm-esc4
# basic prediction with time series only
# and with other features [weather features]
# use one week's data to predict next hour


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

import datetime
from datetime import timedelta
import datetime_utils
# import ha
# import sarima
# import arima
import argparse

import lstm
# import lstm_attn
# import lstm_varying_window
import evaluation

MAX_TIMESTEPS = 336
MIN_TIMESTEPS = 24

TRAINING_STEPS = 3000
LEARNING_RATE = 0.001
TIMESTEPS = 168


class train:
    # TODO: increase window size to 4 weeks
    def __init__(self, raw_df, window = 168):
        self.raw_df = raw_df
        self.train_start_time = '2014-02-01'
        #self.train_end_time = '2018-03-31'
        self.train_end_time = '2018-10-31'
        # set train/test set
        #self.test_start_time = '2018-04-01 00:00:00'
        #self.test_end_time = '2018-04-30 23:00:00'
        self.test_start_time = '2018-11-01 00:00:00'
        self.test_end_time = '2019-04-30 23:00:00'

        # if not speficied, prediction window: use one week's data to predict next hour
        #self.window = datetime.timedelta(hours=24 * 7)
        self.window = datetime.timedelta(hours=window)
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
        # self.grid_list = list(raw_df)


def parse_args():
    parser = argparse.ArgumentParser()
    #                 action="store", help = 'whether to multi-var fairloss. If True, include aga, race, and edu. Otherwise, use race')
    parser.add_argument('-s',   '--suffix',
                     action="store", help = 'save path suffix', default = '')

    parser.add_argument('-use_1d_fea',   type=bool, default=False,
                    action="store", help = 'whether to use 1d features. If use this option, set to True. Otherwise, default False')

    parser.add_argument('-use_latent_fea',   type=bool, default=False,
                        action="store", help = 'whether to use latent features. If use this option, set to True. Otherwise, default False')

    parser.add_argument("-r","--resume_training", type=bool, default=False,
    				help="A boolean value whether or not to resume training from checkpoint")
    parser.add_argument('-t',   '--train_dir',
                     action="store", help = 'training dir containing checkpoints', default = '')
    parser.add_argument('-c',   '--checkpoint',
                     action="store", help = 'checkpoint path (resume training)', default = None)
    parser.add_argument('-p',   '--place',
                     action="store", help = 'city to train on: Seattle or Austin', default = 'Seattle')
    parser.add_argument('-e',   '--epoch',  type=int,
                     action="store", help = 'epochs to train', default = 3000)
    parser.add_argument('-l',   '--learning_rate',  type=float,
                     action="store", help = 'epochs to train', default = 0.001)
    parser.add_argument('-d',   '--encoding_dir',
                     action="store", help = 'dir containing latent representations', default = '')

    

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
    encoding_dir = args.encoding_dir

    use_1d_fea = bool(args.use_1d_fea)
    use_latent_fea = bool(args.use_latent_fea)
    encoding_dir = args.encoding_dir

    print("resume_training: ", resume_training)
    print("training dir path: ", train_dir)
    print("checkpoint: ", checkpoint)
    print("place: ", place)
    print("epochs to train: ", epoch)
    print("start learning rate: ", learning_rate)
    print("use_1d_fea: ", use_1d_fea)
    print("use_latent_fea: ", use_latent_fea)

    if checkpoint is not None:
        checkpoint = train_dir + checkpoint
        print('pick up checkpoint: ', checkpoint)

    globals()['TRAINING_STEPS']  = epoch
    globals()['LEARNING_RATE']  = learning_rate
    print('TRAINING_STEPS: ', TRAINING_STEPS)

    #hourly_grid_timeseries = pd.read_csv('./hourly_grid_1000_timeseries_trail.csv', index_col = 0)
    hourly_grid_timeseries = pd.read_csv('../data_processing/Fremont_bicycle_count_clean_final.csv', index_col = 0)
    hourly_grid_timeseries.index = pd.to_datetime(hourly_grid_timeseries.index)
    hourly_grid_timeseries = pd.DataFrame(hourly_grid_timeseries['total_count'])


    # -------  load extra features --------------------- #
    path_1d = '../data_processing/1d_source_data/'
    if use_1d_fea:
        # 1d
        weather_arr = np.load(path_1d + 'weather_arr_20140201_20190501.npy')
        print('weather_arr.shape: ', weather_arr.shape)
        weather_arr = weather_arr[0,0,0:-24,:]  # until 20190430
        print('weather_arr.shape: ', weather_arr.shape)

        hourly_grid_timeseries['precipitation'] = list(weather_arr[:,0].flatten())
        hourly_grid_timeseries['temperature'] = list(weather_arr[:,1].flatten())
        hourly_grid_timeseries['pressure'] = list(weather_arr[:,2].flatten())

        # hourly_grid_timeseries = np.concatenate([hourly_grid_timeseries,weather_arr], axis=1)
        # print('hourly_grid_timeseries.shape', hourly_grid_timeseries.shape)

    if use_latent_fea:
        latent_rep_path = '/home/ubuntu/CTensor/' + encoding_dir + 'latent_rep/final_lat_rep.npy'
        latent_rep = np.load(latent_rep_path)
        # deprecated: (41616, 1, 32, 20, 1) for v1,  (41616, 32, 20, 1) for v2
        print('latent_rep.shape: ', latent_rep.shape)
        # (45960, 5)
        latent_bridge_rep = latent_rep[:, 11, 8, :]  # the location of fremont bridge
        latent_bridge_rep = latent_bridge_rep[:-24, :]
        latent_df = pd.DataFrame(latent_bridge_rep)
        hourly_grid_timeseries = pd.concat([hourly_grid_timeseries,latent_df], axis=1)
        # hourly_grid_timeseries['precipitation'] = list(weather_arr[:,0].flatten())
        # hourly_grid_timeseries['temperature'] = list(weather_arr[:,1].flatten())
        # hourly_grid_timeseries['pressure'] = list(weather_arr[:,2].flatten())



    print(hourly_grid_timeseries.head())
    print(list(hourly_grid_timeseries))
    # ################## !!!!! ####################################
    # need to specify window size if varying window scheme is used
    train_obj = train(hourly_grid_timeseries,  window = 168)

    if suffix == '':
        save_path =  './bikecount' + '_'  +str(use_1d_fea)
    else:
        save_path = './bikecount'+ '_'  +suffix + '_'  +str(use_1d_fea) +'/'

    if train_dir:
        save_path = train_dir

    # print("training dir: ", train_dir)
    print("save_path: ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save_path = './'


    # lstm
    print('lstm prediction')
    if resume_training == False:
    #lstm_predicted = pd.read_csv(save_path + 'lstm_predicted.csv', index_col=0)
    #lstm_predicted.index = pd.to_datetime(lstm_predicted.index)
        lstm_predicted = lstm.lstm(train_obj,save_path,
                    TIMESTEPS,
               TRAINING_STEPS, LEARNING_RATE).lstm_predicted
        lstm_predicted.to_csv(save_path + 'lstm_predicted.csv')
    else:
        print('resume trainging from : ', train_dir)
        lstm_predicted = lstm.lstm(train_obj,save_path,
                    TIMESTEPS,
               TRAINING_STEPS, LEARNING_RATE,
               False, checkpoint, True, train_dir).lstm_predicted
        lstm_predicted.to_csv(save_path + 'lstm_predicted.csv')

    # eval_obj6 = evaluation.evaluation(train_obj.test_df, lstm_predicted)
    # print('rmse for lstm: ',eval_obj6.rmse_val)
    # print('mae for lstm: ', eval_obj6.mae_val)



    # lstm with VARYING window
    # print('lstm prediction with VARYING window')
    # #lstm_predicted = pd.read_csv(save_path + 'lstm_predicted.csv', index_col=0)
    # #lstm_predicted.index = pd.to_datetime(lstm_predicted.index)
    # lstm_varying_predicted = lstm_varying_window.lstm(train_obj).lstm_predicted
    # lstm_varying_predicted.to_csv(save_path + 'lstm_varying_predicted.csv')
    # eval_obj8 = evaluation.evaluation(train_obj.test_df, lstm_varying_predicted)
    # print('rmse for lstm: ',eval_obj8.rmse_val)
    # print('mae for lstm: ', eval_obj8.mae_val)


    # attention lstm
    #print('lstm prediction with ATTENTION')
    # lstm_predicted = pd.read_csv(save_path + 'lstm_predicted.csv', index_col=0)
    # lstm_predicted.index = pd.to_datetime(lstm_predicted.index)
    #lstm_attn_predicted = lstm_attn.lstm(train_obj).lstm_predicted
    #lstm_attn_predicted.to_csv(save_path + 'lstm_attn_predicted.csv')
    #eval_obj7 = evaluation.evaluation(train_obj.test_df, lstm_attn_predicted)
    #print('rmse for lstm: ',eval_obj7.rmse_val)
    #print('mae for lstm: ', eval_obj7.mae_val)



    # with open('lstm_bikecount_output.txt', 'w') as the_file:
    #     the_file.write('rmse for lstm\n')
    #     the_file.write(str(eval_obj6.rmse_val))
    #     the_file.write('mae for lstm\n')
    #     the_file.write(str(eval_obj6.mae_val))
    #     the_file.close()




if __name__ == '__main__':
    main()
