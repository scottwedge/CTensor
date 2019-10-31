import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_absolute_error

# added
from sklearn.metrics import mean_squared_error
import datetime
import datetime_utils

# TODO: currently, for SARIMA and ARIMA, if the training series is all 0,
# the predicted value will be set to zero, so in evaluation, these were taken
# into account. Would be more accurate if these were teased out from evaluation
class evaluation(object):
    def __init__(self, gt_df,pred_df, demo_raw = None):
        self.gt_df = gt_df
        self.pred_df = pred_df
        self.rmse_val = self.rmse()
        self.mae_val = self.mae()
        self.mape_val = self.mape()
        self.demo_raw = demo_raw
        # self.mae_df = self.mae_over_city()
        # self.rmse_df = self.rmse_over_city()


    def rmse(self):
        #gt_df = gt_df[gt_df.index == pred_df.index]
        mse = 0
        pred_df = self.pred_df.dropna()
        # df1 = df1[~df1.index.isin(df2.index)]
        gt_df = self.gt_df[self.gt_df.index.isin(pred_df.index)]
        for fea in list(gt_df):
            y_forecasted = pred_df[fea]
            y_truth = gt_df[fea]
            # Compute the mean square error
            #mse =+ ((y_forecasted - y_truth) ** 2).mean(skipna = True)
            mse += ((y_forecasted - y_truth) ** 2).sum(skipna = True)
        mse = float(mse)/ (len(list(gt_df)) * len(gt_df))
        #print('The Mean Squared Error of our forecasts is {}'.format(mse))
        return math.sqrt(mse)



    def mae(self):
        #gt_df = gt_df[gt_df.index == pred_df.index]
        pred_df = self.pred_df.dropna()
        # df1 = df1[~df1.index.isin(df2.index)]
        gt_df = self.gt_df[self.gt_df.index.isin(pred_df.index)]
        mae = 0
        for fea in list(gt_df):
            y_forecasted = pred_df[fea]
            y_truth = gt_df[fea]

            # Compute the mean square error
            mae += mean_absolute_error(y_truth, y_forecasted) * len(y_truth)
        mae = float(mae)/ (len(list(gt_df)) * len(gt_df))
        print('The Mean absolute error {}'.format(mae))
        return mae



    def mape(self):
        #gt_df = gt_df[gt_df.index == pred_df.index]
        pred_df = self.pred_df.dropna()
        # df1 = df1[~df1.index.isin(df2.index)]
        gt_df = self.gt_df[self.gt_df.index.isin(pred_df.index)]
        mape = 0
        for fea in list(gt_df):
            y_forecasted = pred_df[fea].to_numpy()
            y_truth = gt_df[fea].to_numpy()
            # Compute the mean square error
            # mape += mean_absolute_error(y_truth, y_forecasted) * len(y_truth)

            mape += np.nansum(np.divide(np.absolute(y_truth - y_forecasted), y_truth))
        mape = float(mape)/ (len(list(gt_df)) * len(gt_df))
        print('The Mean absolute Percent error {}'.format(mape))
        return mape




        # return a dataframe
    #     grid_num1  grid_num2 ...
    #  0    mae        mae
    def mae_over_city(self):
        pred_df = self.pred_df.dropna()
        # df1 = df1[~df1.index.isin(df2.index)]
        gt_df = self.gt_df[self.gt_df.index.isin(pred_df.index)]
        mae_df = pd.DataFrame(0, columns = list(gt_df), index = ['mae'])

        for fea in list(gt_df):
            mae = 0
            y_forecasted = pred_df[fea]
            y_truth = gt_df[fea]

            # Compute the mean square error
            mae = mean_absolute_error(y_truth, y_forecasted)
            mae_df.loc['mae', fea] = mae

        #mae = float(mae)/ (len(list(gt_df)) * len(gt_df))
        #print('The Mean absolute error {}'.format(mae))
        return mae_df


    # return a dataframe
    #     grid_num1  grid_num2 ...
    #  0    mae        mae
    def rmse_over_city(self):
        pred_df = self.pred_df.dropna()
        # df1 = df1[~df1.index.isin(df2.index)]
        gt_df = self.gt_df[self.gt_df.index.isin(pred_df.index)]
        rmse_df = pd.DataFrame(0, columns = list(gt_df), index = ['rmse'])

        for fea in list(gt_df):
            mae = 0
            y_forecasted = pred_df[fea]
            y_truth = gt_df[fea]

            # Compute the mean square error
            rmse  = math.sqrt(mean_squared_error(y_truth, y_forecasted))

            rmse_df.loc['rmse', fea] = rmse

        #mae = float(mae)/ (len(list(gt_df)) * len(gt_df))
        #print('The Mean absolute error {}'.format(mae))
        return rmse_df


    # TODO: pos res diff per capita is wrong!
    '''
    G2: under-represented group
    - G+: # of occurrances where yi_hat > yi

    mean difference: mean (yi_hat in G1 ) - mean(yi_hat in G2)
    residual difference: mean( yi_hat - yi) in G1+) - mean( yi_hat - yi) in G2+)
    absolute residual difference: mean( |yi_hat - yi|) in G1+) - mean( |yi_hat - yi|) in G2+)
    positive residual difference: mean( max (0, yi_hat - yi) in G1+) - mean( max (0, yi_hat - yi) in G2+)


    # does not divide by # of grids in each group

    absolute res diff per captia: ( |yi_hat - yi| in G1 ) / pop_G1 -  (|yi_hat -yi| in G2)  / pop_G2
    mean res diff per captia: (yi_hat - yi in G1 ) / pop_G1 -  (yi_hat -yi in G2)  / pop_G2
    pos res diff per capita: ( max (0, yi_hat - yi) in G1+) / pop_G1 - mean( max (0, yi_hat - yi) in G2+)/ pop_G2
    mean diff per capita:  (yi_hat in G1 ) / pop_G1 -  (yi_hat in G2)  / pop_G2



    groups: 'bi_caucasian','bi_age', 'bi_high_incm','bi_edu_univ','bi_nocar_hh'

    output:

                 'bi_caucasian','bi_age', 'bi_high_incm','bi_edu_univ','bi_nocar_hh'
    mean_diff
    pos_res_diff
    neg_res_diff
    '''
    def group_difference(self):
        pred_df = self.pred_df.dropna()
        # df1 = df1[~df1.index.isin(df2.index)]
        gt_df = self.gt_df[self.gt_df.index.isin(pred_df.index)]
        group_list = ['bi_caucasian','bi_age', 'bi_high_incm','bi_edu_univ','bi_nocar_hh']

        diff_df = pd.DataFrame(0, columns = group_list, index = ['mean_diff','pos_res_diff','neg_res_diff', 'res_diff',
                                                                'mean_diff_percap', 'res_diff_percap', 'pos_res_diff_percap',
                                                               'neg_res_diff_percap', 'abs_res_diff_percap' ])

        # iterate through caucasian, age, income....
        for group in group_list:
            # g1: a list of grid num that belongs to group 1
            g1 =  self.demo_raw[self.demo_raw[group] == 1]['pos'].tolist()
            g2 = self.demo_raw[self.demo_raw[group] == -1]['pos'].tolist()

            group1_df =  pred_df[pred_df.columns.intersection(g1)]
            group2_df = pred_df[pred_df.columns.intersection(g2)]

            diff_df.loc['mean_diff', group] = group1_df.values.mean()- group2_df.values.mean()

            # mean_diff_percap: (yi_hat in G1 ) / pop_G1 -  (yi_hat in G2)  / pop_G2
            # sum along grids, mean over time
            bike_g1 = group1_df.values.sum() / float(len(pred_df))
            bike_g2 = group2_df.values.sum()/ float(len(pred_df))

            g1_pop = float(self.demo_raw[self.demo_raw[group] == 1]['normalized_pop'].sum())
            g2_pop = float(self.demo_raw[self.demo_raw[group] == -1]['normalized_pop'].sum())

            diff_df.loc['mean_diff_percap', group] = bike_g1 /g1_pop -bike_g2 /g2_pop



            g1_pos_res = 0
            g1_neg_res = 0
            g1_res = 0

            g1_abs_res = 0
            g1_res_pergrid = 0
            g1_pos_res_pergrid = 0
            g1_neg_res_pergrid = 0
            # iterate through G1 grid num
            for fea in list(group1_df):
                # if fea is in G1 or G2
                # G1 being the overrepresented
                G1_plus = 0


                for idx in range(len(group1_df)):
                    pred_cell = pred_df[fea].iloc[idx]
                    gt_cell = gt_df[fea].iloc[idx]

                    g1_pos_res += max((pred_cell - gt_cell), 0)
                    g1_neg_res += max((gt_cell - pred_cell), 0)
                    g1_res += pred_cell - gt_cell

                    g1_abs_res += abs(pred_cell - gt_cell)

                    if pred_cell > gt_cell:
                        G1_plus+=1
                if G1_plus == 0:
                    g1_pos_res_pergrid = 99999
                    g1_neg_res_pergrid = 99999
                else:
                    g1_pos_res_pergrid = g1_pos_res / float(G1_plus)
                    g1_neg_res_pergrid = g1_neg_res / float(G1_plus)
                    g1_res_pergrid = g1_res / float(len(group1_df))

            g2_pos_res = 0
            g2_neg_res = 0
            g2_res = 0

            g2_abs_res = 0
            g2_res_pergrid = 0
            # iterate through G2 grid num
            for fea in list(group2_df):
                # G1 being the overrepresented
                G2_plus = 0
                pos_res = 0
                neg_res = 0
                for idx in range(len(group2_df)):
                    pred_cell = pred_df[fea].iloc[idx]
                    gt_cell = gt_df[fea].iloc[idx]

                    g2_pos_res += max((pred_cell - gt_cell), 0)
                    g2_neg_res += max((gt_cell - pred_cell), 0)
                    g2_res += pred_cell - gt_cell

                    g2_abs_res += abs(pred_cell - gt_cell)

                    if pred_cell < gt_cell:
                        G2_plus+=1

                if G2_plus == 0:
                    g2_pos_res_pergrid = 99999
                    g2_neg_res_pergrid = 99999
                else:
                    g2_pos_res_pergrid = g2_pos_res / float(G2_plus)
                    g2_neg_res_pergrid = g2_neg_res / float(G2_plus)
                    g2_res_pergrid = g2_res / float(len(group2_df))



            diff_df.loc['pos_res_diff', group] = g1_pos_res_pergrid- g2_pos_res_pergrid
            diff_df.loc['neg_res_diff', group] = g1_neg_res_pergrid- g2_neg_res_pergrid
            diff_df.loc['res_diff', group] = g1_res_pergrid- g2_res_pergrid

            diff_df.loc['abs_res_diff_percap', group] = g1_abs_res/g1_pop - g2_abs_res/g2_pop
            diff_df.loc['res_diff_percap', group] = g1_res/g1_pop - g2_res/g2_pop

            diff_df.loc['pos_res_diff_percap', group] = g1_pos_res/g1_pop - g2_pos_res/g2_pop
            diff_df.loc['neg_res_diff_percap', group] = g1_neg_res/g1_pop - g2_neg_res/g2_pop


        return diff_df


        # sum (Et (y_hat) * white_perc )  / pop_white - sum (Et (y_hat) * nonwhite_perc )/ pop_nonwhite
    def individual_difference(self):
        pred_df = self.pred_df.dropna()
        # df1 = df1[~df1.index.isin(df2.index)]
        gt_df = self.gt_df[self.gt_df.index.isin(pred_df.index)]
       # group_list = ['bi_caucasian','bi_age', 'bi_high_incm','bi_edu_univ','bi_nocar_hh']

        #diff_df = pd.DataFrame(0, columns = group_list, index = ['mean_diff_percap'])
        gt_mean_df = pd.DataFrame(pred_df.mean(axis=0)).T

        # mean trip start is hourly mean per person over all time for a group.
        # mean trip start = sum(# of bikes) in group1 of all time / pop in group1
        gt_equity_cols = ['overall','caucasian', 'non_caucasian', 'senior','young',
                         'high_edu','low_edu']
        gt_equity_df = pd.DataFrame(0,  index=['mean_tripstart', 'pop'], columns=gt_equity_cols)

       # iterating through all grids
        for idx, row in self.demo_raw.iterrows():
            grid_num = row['pos']
            if(pd.isnull(row['asian_pop'])):
                continue
            gt_equity_df.loc['mean_tripstart','overall'] += gt_mean_df[grid_num][0]
            #gt_equity_df.loc['num_grid','overall'] += 1
            gt_equity_df.loc['pop','overall'] += row['normalized_pop']

    #         if row['bi_caucasian'] == 1:
            gt_equity_df.loc['mean_tripstart','caucasian'] += gt_mean_df[grid_num][0] * row['white_pop']
            #gt_equity_df.loc['num_grid','caucasian'] += 1
            gt_equity_df.loc['pop','caucasian'] += row['normalized_pop']* row['white_pop']

            #if row['bi_caucasian'] == -1:
            gt_equity_df.loc['mean_tripstart','non_caucasian'] += gt_mean_df[grid_num][0] * (100 - row['white_pop'])
                #gt_equity_df.loc['num_grid','non_caucasian'] += 1
            gt_equity_df.loc['pop','non_caucasian'] += row['normalized_pop']* (100 - row['white_pop'])

            #if row['bi_age'] == 1:
            gt_equity_df.loc['mean_tripstart','young'] += gt_mean_df[grid_num][0]* row['age65_under']
                # gt_equity_df.loc['num_grid','young'] += 1
            gt_equity_df.loc['pop','young'] += row['normalized_pop']* row['age65_under']

            #if row['bi_age'] == -1:
            gt_equity_df.loc['mean_tripstart','senior'] += gt_mean_df[grid_num][0]* row['age65']
                 #gt_equity_df.loc['num_grid','senior'] += 1
            gt_equity_df.loc['pop','senior'] += row['normalized_pop']* row['age65']

            #if row['bi_edu_univ'] == 1:
            gt_equity_df.loc['mean_tripstart','high_edu'] += gt_mean_df[grid_num][0] *  row['edu_uni']
                 #gt_equity_df.loc['num_grid','high_edu'] += 1
            gt_equity_df.loc['pop','high_edu'] += row['normalized_pop']*  row['edu_uni']
            #if row['bi_edu_univ'] == -1:
            gt_equity_df.loc['mean_tripstart','low_edu'] += gt_mean_df[grid_num][0] * (100 -row['edu_uni'])
                 #gt_equity_df.loc['num_grid','low_edu'] += 1
            gt_equity_df.loc['pop','low_edu'] += row['normalized_pop'] * (100 -row['edu_uni'])


        #gt_equity_df.loc['mean_tripstart','caucasian'] = gt_equity_df['caucasian']['mean_tripstart']/float(gt_equity_df['caucasian']['num_grid'])
        gt_equity_df.loc['mean_tripstart','caucasian'] = gt_equity_df['caucasian']['mean_tripstart']/float(gt_equity_df['caucasian']['pop'])
        gt_equity_df.loc['mean_tripstart','non_caucasian'] = gt_equity_df['non_caucasian']['mean_tripstart']/float(gt_equity_df['non_caucasian']['pop'])

        gt_equity_df.loc['mean_tripstart','senior'] = gt_equity_df['senior']['mean_tripstart']/float(gt_equity_df['senior']['pop'])
        gt_equity_df.loc['mean_tripstart','young'] = gt_equity_df['young']['mean_tripstart']/float(gt_equity_df['young']['pop'])

    #     gt_equity_df.loc['mean_tripstart','high_incm'] = gt_equity_df['high_incm']['mean_tripstart']/float(gt_equity_df['high_incm']['pop'])
    #     gt_equity_df.loc['mean_tripstart','low_incm'] = gt_equity_df['low_incm']['mean_tripstart']/float(gt_equity_df['low_incm']['pop'])

        gt_equity_df.loc['mean_tripstart','high_edu'] = gt_equity_df['high_edu']['mean_tripstart']/float(gt_equity_df['high_edu']['pop'])
        gt_equity_df.loc['mean_tripstart','low_edu'] = gt_equity_df['low_edu']['mean_tripstart']/float(gt_equity_df['low_edu']['pop'])

    #     gt_equity_df.loc['mean_tripstart','fewer_car'] = gt_equity_df['fewer_car']['mean_tripstart']/float(gt_equity_df['fewer_car']['pop'])
    #     gt_equity_df.loc['mean_tripstart','more_car'] = gt_equity_df['more_car']['mean_tripstart']/float(gt_equity_df['more_car']['pop'])

        gt_equity_df.loc['mean_tripstart','overall'] = gt_equity_df['overall']['mean_tripstart']/float(gt_equity_df['overall']['pop'])


        gt_mean_bike_perperson_diff = pd.DataFrame(0, index = ['diff'], columns=['caucasian_non_caucasian',
                                                                             'young_senior',
                                                          'high_edu_low_edu'])

        gt_mean_bike_perperson_diff.loc['diff','caucasian_non_caucasian'] = gt_equity_df['caucasian']['mean_tripstart'] -gt_equity_df['non_caucasian']['mean_tripstart']
        gt_mean_bike_perperson_diff.loc['diff','young_senior'] = gt_equity_df['young']['mean_tripstart'] -gt_equity_df['senior']['mean_tripstart']

       # gt_mean_bike_perperson_diff.loc['diff','high_incm_low_incm'] = gt_equity_df['high_incm']['mean_tripstart'] -gt_equity_df['low_incm']['mean_tripstart']

        gt_mean_bike_perperson_diff.loc['diff','high_edu_low_edu'] = gt_equity_df['high_edu']['mean_tripstart'] -gt_equity_df['low_edu']['mean_tripstart']
        #gt_mean_bike_perperson_diff.loc['diff','more_car_fewer_car'] = gt_equity_df['more_car']['mean_tripstart'] -gt_equity_df['fewer_car']['mean_tripstart']

#         return gt_equity_df,gt_mean_bike_perperson_diff

        return gt_mean_bike_perperson_diff
