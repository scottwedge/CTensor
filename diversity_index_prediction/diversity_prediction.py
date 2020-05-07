# updated May 6, 2020
# interpolate diversity index defined by USA today
# based on 2018 demographic data
# with pertinent features and city tensors
# method: Lasso regression

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
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

HEIGHT = 32
WIDTH = 20

def lasso(input, feature_set):
    target_var = ['diversity_index']
    X = input[feature_set]
    y = input[target_var]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    clf = linear_model.Lasso(alpha= 0.01)
    clf.fit(X_train, y_train)
    for i in range(0, len(feature_set)):
        print(feature_set[i], clf.coef_[i])

    train_score=clf.score(X_train, y_train)
    print(train_score)

    test_score=clf.score(X_test,y_test)
    print(test_score)

    return train_score, test_score



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',   '--suffix',
                     action="store", help = 'save path suffix', default = '')
    parser.add_argument('-e',   '--epoch',  type=int,
                     action="store", help = 'epochs to train', default = 200)
    parser.add_argument('-l',   '--learning_rate',  type=float,
                     action="store", help = 'epochs to train', default = 0.003)
    parser.add_argument('-d',   '--encoding_dir',
                     action="store", help = 'dir containing latent representations', default = '')

    #parser.add_argument('-a','--use_1d_fea', type=bool, default=True,
     #                action="store", help = 'whether to use 1d features')
    #parser.add_argument('-b','--use_2d_fea', type=bool, default=True,
     #                 action="store", help = 'whether to use 2d features')
    # parser.add_argument('city_stat', help = 'city-wide demographic data in csv format')
    return parser.parse_args()




def main():
    args = parse_args()
    suffix = args.suffix
    encoding_dir = args.encoding_dir
    ########## loading data ###############################################
    print('loading latent representation')
    latent_rep_path = '/home/ubuntu/CTensor/' + encoding_dir + 'latent_rep/final_lat_rep.npy'
    latent_rep = np.load(latent_rep_path)
    print('latent_rep.shape: ', latent_rep.shape)  # should be [42240, 32, 20, 3]
    num_latent_rep = latent_rep.shape[-1]
    # latent_rep =latent_rep.reshape((45960, 32, 20, 5))
    # take 2018 data from latent rep
    latent_series = latent_rep[34344:43104,  :,:,:]
    latent_series_mean = np.mean(latent_series, axis = 0)

    # load grid csv
    grid = pd.read_csv('../auxillary_data/grid_32_20.csv',index_col = 0)
    latent_series_df = grid[['pos','row', 'col', 'grid_area', 'geometry']].copy()

    for c in range(num_latent_rep): # dimension of latent represenation
        colname =  'latent_val_' +  str(c)  # 'latent_val_0'
        for i in range(32): # col
            for j in range(20):  # row
                pos = str(j)+ '_' + str(i)
                # print('pos', pos)
                idx = latent_series_df[latent_series_df['pos'] == pos].index
                # print(idx)
                latent_series_df.loc[idx, colname] = latent_series_mean[i, j, c]

    # read diversity groud truth
    print('combining diversity_df and latent rep')
    diversity_df = pd.read_csv('diversity_index_clean.csv', index_col = 0)
    # combine diversity_df and latent representations
    combined_df = pd.merge(latent_series_df, diversity_df, how='inner',  on = ['pos'])
    print(combined_df.head())

    if suffix == '':
        save_path =  './diversity' + '/'
    else:
        save_path = './diversity'+ '_'+ suffix  +'/'

    print("save_path: ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ###################  REGRESSION ############################
    # lasso regression
    # spatial features only
    print('spatial features only')
    spatial_feature_set = [ 'row_x',  'col_x']
    spatial_train_score, spatial_test_score = lasso(combined_df, spatial_feature_set)

    # latent rep only
    print('latent rep only')
    latent_rep_set = ['latent_val_' +  str(c) for c in range(num_latent_rep)]
    latrep_train_score, latrep_test_score = lasso(combined_df, latent_rep_set)

    # spatial + latent rep
    print('spatial + latent rep')
    spatial_latrep_set = latent_rep_set+ spatial_feature_set
    spatial_latrep_train_score, spatial_latrep_test_score = lasso(combined_df, spatial_latrep_set)

    # oracle feature_set
    print('oracle feature_set')
    oracle_set = [ 'edu_uni',  'edu_high', 'poverty_po_norm', 'no_car_hh','age65', 'row_x',  'col_x' ]
    oracle_train_score, oracle_test_score = lasso(combined_df, oracle_set)

    # oracle + latent rep
    print(' oracle + latent rep')
    oracle_latrep_set = oracle_set + latent_rep_set
    oracle_latrep_train_score, oracle_latrep_test_score = lasso(combined_df, oracle_latrep_set)

    # save results to csv
    indexes = ['spatial', 'latent_rep', 'spatial_latrep', 'oracle', 'oracle_latrep']
    result_df = pd.DataFrame(None, index =indexes,
                    columns= ['train_score', 'test_score'])
    result_df.loc['spatial', 'train_score'] = spatial_train_score
    result_df.loc['spatial', 'test_score'] = spatial_test_score

    result_df.loc['latent_rep', 'train_score'] = latrep_train_score
    result_df.loc['latent_rep', 'test_score'] = latrep_test_score

    result_df.loc['spatial_latrep', 'train_score'] = spatial_latrep_train_score
    result_df.loc['spatial_latrep', 'test_score'] = spatial_latrep_test_score

    result_df.loc['oracle', 'train_score'] = oracle_train_score
    result_df.loc['oracle', 'test_score'] = oracle_test_score

    result_df.loc['oracle_latrep', 'train_score'] = oracle_latrep_train_score
    result_df.loc['oracle_latrep', 'test_score'] = oracle_latrep_test_score

    result_df.to_csv(save_path+ 'result_df.csv')

    timer = str(time.time())
    txt_name = save_path + 'diversity' +   timer + '.txt'
    with open(txt_name, 'w') as the_file:
        the_file.write('encoding dir\n')
        the_file.write(str(encoding_dir) + '\n')

        the_file.write('spatial features only: train score, test score: \n')
        the_file.write(str(spatial_train_score) + '\n')
        the_file.write(str(spatial_test_score) + '\n')

        the_file.write('latent rep only only: train score, test score: \n')
        the_file.write(str(latent_rep_set) + '\n')
        the_file.write(str(latrep_test_score) + '\n')

        the_file.write('latent rep only only: train score, test score: \n')
        the_file.write(str(latent_rep_set) + '\n')
        the_file.write(str(latrep_test_score) + '\n')


        the_file.close()







if __name__ == '__main__':
    main()
