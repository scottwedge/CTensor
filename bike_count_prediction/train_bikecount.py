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
import ha
import sarima
import arima

import lstm
import lstm_attn
import lstm_varying_window
import evaluation

MAX_TIMESTEPS = 336
MIN_TIMESTEPS = 24

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




def main():
    #hourly_grid_timeseries = pd.read_csv('./hourly_grid_1000_timeseries_trail.csv', index_col = 0)
    hourly_grid_timeseries = pd.read_csv('../data_processing/Fremont_bicycle_count_clean_final.csv', index_col = 0)
    hourly_grid_timeseries.index = pd.to_datetime(hourly_grid_timeseries.index)

    # ################## !!!!! ####################################
    # need to specify window size if varying window scheme is used
    train_obj = train(hourly_grid_timeseries,  window = 168)
    save_path = './'


    # lstm
    print('lstm prediction')
    #lstm_predicted = pd.read_csv(save_path + 'lstm_predicted.csv', index_col=0)
    #lstm_predicted.index = pd.to_datetime(lstm_predicted.index)
    lstm_predicted = lstm.lstm(train_obj).lstm_predicted
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
