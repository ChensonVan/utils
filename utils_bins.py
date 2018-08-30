import pandas as pd
import numpy as np
import re, math, hashlib, json
from sklearn.utils.multiclass import type_of_target

# import packages
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model  import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import f_regression
import math
from scipy import stats
from sklearn.utils.multiclass import type_of_target

# import Orange
import random
from math import *
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf

import traceback

# from utils_woe import WOE
# import utils.utils_model_monitor as umm

class WOE:
    def __init__(self):
        self._WOE_MIN = -20
        self._WOE_MAX = 20

    def woe(self, X, y, event=1):
        '''
        Calculate woe of each feature category and information value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable which should be binary
        :param event: value of binary stands for the event to predict
        :return: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
                 numpy array of information value of each feature
        '''
        self.check_target_binary(y)
        X1 = self.feature_discretion(X)

        res_woe = []
        res_iv = []
        for i in range(0, X1.shape[-1]):
            x = X1[:, i]
            woe_dict, iv1 = self.woe_single_x(x, y, event)
            res_woe.append(woe_dict)
            res_iv.append(iv1)
        return np.array(res_woe), np.array(res_iv)

    def woe_single_x(self, x, y, event=1):
        '''
        calculate woe and information for a single feature
        :param x: 1-D numpy starnds for single feature
        :param y: 1-D numpy array target variable
        :param event: value of binary stands for the event to predict
        :return: dictionary contains woe values for categories of this feature
                 information value of this feature
        '''
        self.check_target_binary(y)

        event_total, non_event_total = self.count_binary(y, event=event)
        x_labels = pd.Categorical(x).categories.values
        woe_dict = {}
        iv = 0
        for x1 in x_labels:
            y1 = y[np.where(x == x1)[0]]
            event_count, non_event_count = self.count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe1 = self._WOE_MIN
            elif rate_non_event == 0:
                woe1 = self._WOE_MAX
            else:
                woe1 = math.log(rate_event / rate_non_event)
            woe_dict[x1] = woe1
            iv += (rate_event - rate_non_event) * woe1
        return woe_dict, iv

    def woe_replace(self, X, woe_arr):
        '''
        replace the explanatory feature categories with its woe value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param woe_arr: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
        :return: the new numpy array in which woe values filled
        '''
        if X.shape[-1] != woe_arr.shape[-1]:
            raise ValueError('WOE dict array length must be equal with features length')

        res = np.copy(X).astype(float)
        idx = 0
        for woe_dict in woe_arr:
            for k in woe_dict.keys():
                woe = woe_dict[k]
                res[:, idx][np.where(res[:, idx] == k)[0]] = woe * 1.0
            idx += 1

        return res

    def combined_iv(self, X, y, masks, event=1):
        '''
        calcute the information vlaue of combination features
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable
        :param masks: 1-D numpy array of masks stands for which features are included in combination,
                      e.g. np.array([0,0,1,1,1,0,0,0,0,0,1]), the length should be same as features length
        :param event: value of binary stands for the event to predict
        :return: woe dictionary and information value of combined features
        '''
        if masks.shape[-1] != X.shape[-1]:
            raise ValueError('Masks array length must be equal with features length')

        x = X[:, np.where(masks == 1)[0]]
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(self.combine(x[i, :]))

        dumy = np.array(tmp)
        # dumy_labels = np.unique(dumy)
        woe, iv = self.woe_single_x(dumy, y, event)
        return woe, iv


    def combine(self, list):
        res = ''
        for item in list:
            res += str(item)
        return res


    def count_binary(self, a, event=1):
        event_count = (a == event).sum()
        non_event_count = a.shape[-1] - event_count
        return event_count, non_event_count


    def check_target_binary(self, y):
        '''
        check if the target variable is binary, raise error if not.
        :param y:
        :return:
        '''
        y_type = type_of_target(y)
        if y_type not in ['binary']:
            raise ValueError('Label type must be binary')


    def feature_discretion(self, X):
        '''
        Discrete the continuous features of input data X, and keep other features unchanged.
        :param X : numpy array
        :return: the numpy array in which all continuous features are discreted
        '''
        temp = []
        for i in range(0, X.shape[-1]):
            x = X[:, i]
            x_type = type_of_target(x)
            if x_type == 'continuous':
                x1 = self.discrete(x)
                temp.append(x1)
            else:
                temp.append(x)
        return np.array(temp).T


    def discrete(self, x):
        '''
        Discrete the input 1-D numpy array using 5 equal percentiles
        :param x: 1-D numpy array
        :return: discreted 1-D numpy array
        '''
        res = np.array([0] * x.shape[-1], dtype=int)
        for i in range(5):
            point1 = stats.scoreatpercentile(x, i * 20)
            point2 = stats.scoreatpercentile(x, (i + 1) * 20)
            x1 = x[np.where((x >= point1) & (x <= point2))]
            mask = np.in1d(x, x1)
            res[mask] = (i + 1)
        return res


    @property
    def WOE_MIN(self):
        return self._WOE_MIN


    @WOE_MIN.setter
    def WOE_MIN(self, woe_min):
        self._WOE_MIN = woe_min


    @property
    def WOE_MAX(self):
        return self._WOE_MAX


    @WOE_MAX.setter
    def WOE_MAX(self, woe_max):
        self._WOE_MAX = woe_max



class Bins(WOE):
    def __init__(self):
        self.RANDOM_STATE = 2018
        self._WOE_MIN = -20
        self._WOE_MAX = 20


    def get_cut_points_by_tree(self, x, y, criterion='gini', max_depth=3, min_samples_leaf=0.01, max_leaf_nodes=None, random_state=2018):
        '''
        根据决策树选出cut_points
        '''
        if type_of_target(y) == 'binary':
            clf = DecisionTreeClassifier(
                                       # criterion=criterion,
                                        max_depth=max_depth, 
                                        min_samples_leaf=min_samples_leaf, 
                                        max_leaf_nodes=max_leaf_nodes, 
                                        random_state=random_state)
        else:
            clf = DecisionTreeRegressor(
                                        # criterion=criterion,
                                        max_depth=max_depth, 
                                        min_samples_leaf=min_samples_leaf, 
                                        max_leaf_nodes=max_leaf_nodes, 
                                        random_state=random_state)

        clf.fit(np.array(x).reshape(-1, 1), np.array(y))
        th = clf.tree_.threshold
        fea = clf.tree_.feature
        # -2 代表的是
        return sorted(th[np.where(fea != -2)])


    def get_cut_points_by_monotonic(self, x, y, num_of_bins=10):
        x, y = pd.Series(x), pd.Series(y)
        x_notnull = x[pd.notnull(x)]
        y_notnull = y[pd.notnull(x)]
        r = 0
        while np.abs(r) < 1:
            d1 = pd.DataFrame({"x": x_notnull, "y": y_notnull, 
                               "Bucket": pd.qcut(x_notnull, num_of_bins, duplicates='drop')})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2['x'].mean(), d2['y'].mean())
            num_of_bins -= 1
        # d3 = pd.DataFrame(d2['x'].min(), columns=['min_x'])
        # print(d2['x'].min().tolist()[:1] + d2['x'].max().tolist())
        # return d2['x'].min().tolist()[:1] + d2['x'].max().tolist()
        # d3['max_x'] = d2['x'].max()
        # d3['min_x'] = d2['x'].min()
        # d3['sum_y'] = d2.sum().y
        # d3['count_y'] = d2.count().y
        # d3['mean_y'] = d2.mean().y
        # d4 = (d3.sort_index(by='min_x')).reset_index(drop=True)
        # return d4
        return d2['x'].min().tolist()[:1] + d2['x'].max().tolist()


    def get_cut_points_by_freq(self, x, num_of_bins=10, precision=4):
        interval = 100 / num_of_bins
        cp = sorted(set(np.percentile(x,  i * interval) for i in range(num_of_bins + 1)))
        cp = np.round(cp, precision).tolist()
        return sorted(set(cp))


    def get_cut_points_by_interval(self, x, num_of_bins=10, precision=4):
        cp = pd.cut(x, num_of_bins, retbins=True)[1]
        return np.round(cp, precision).tolist()



    def bins(self, x, y=None, method='tree', num_of_bins=10, cut_points=None, precision=4):
        if cut_points == None:
            if method == 'tree':
                cut_points = self.get_cut_points_by_tree(x, y)
                
            elif method == 'monotonic':
                cut_points = self.get_cut_points_by_monotonic(x, y, num_of_bins=num_of_bins)

            elif method == 'freq':
                cut_points = self.get_cut_points_by_freq(x, num_of_bins=num_of_bins, precision=precision)

            elif method == 'interval':
                cut_points = self.get_cut_points_by_interval(x, num_of_bins=num_of_bins, precision=precision)

            else:
                raise Exception('INFO : method must be in (tree, freq, interval).')
        
            # print('cut_points1 :', cut_points)
            cut_points = [-np.inf] + cut_points[1:-1] + [np.inf]
            # print('cut_points2 :', cut_points)

        df = pd.DataFrame({'x': x, 
                           'y': y, 
                           'bucket': pd.cut(x, cut_points, include_lowest=True, precision=precision)})
        return df[['x', 'y', 'bucket']]    
    


    def bucket_describe(self, x, y, df_bucket):
        """
        x: 1-D series
        y: 1-D series
        bucket: bucket
        """
        WOE_dic, IV = self.woe_single_x(df_bucket.bucket, df_bucket.y, event=1)

        df_describe = pd.DataFrame()
        df_describe_total = pd.DataFrame()
        
        df_tmp = df_bucket.groupby('bucket', as_index=True)
        df_describe['min_x'] = df_tmp['x'].min()
        df_describe['max_x'] = df_tmp['x'].max()
        df_describe['sum_y'] = df_tmp['y'].sum()
        df_describe['count_y'] = df_tmp['y'].count()
        df_describe['mean_y'] = df_tmp['y'].mean()
        df_describe['bucket'] = pd.Categorical(df_bucket["bucket"]).categories.values
        df_describe['var_name'] = [x.name for i in pd.Categorical(
                df_bucket['bucket']).categories.values]
        df_describe['WOE'] = list(WOE_dic.values())
        df_describe['IV'] = IV
        
        df_describe_total['min_x'] = x.min()
        df_describe_total['max_x'] = x.max()
        df_describe_total['sum_y'] = y.sum()
        df_describe_total['count_y'] = y.count()
        df_describe_total['mean_y'] = y.mean()
        df_describe_total['bucket'] = '_TOTAL_'
        df_describe_total['var_name'] = x.name
        df_describe_total['WOE'] = np.nan
        df_describe_total['IV'] = IV
        
        tmp = pd.concat([df_describe_total, df_describe], axis=0)
        return tmp[['var_name', 'WOE', 'IV', 'bucket', 'min_x', 'max_x', 'sum_y', 'count_y', 'mean_y']]
    
    
    def get_iv(self, x, y):
        from pandas.api.types import is_numeric_dtype
        x_notnull = x[x.notnull()]
        y_notnull = y[x.notnull()]
 
        try:
            if is_numeric_dtype(x):  
                cut_points = self.get_cut_points_by_tree(x_notnull, y_notnull)
                if not cut_points:
                    cut_points = self.get_cut_points_by_freq(x_notnull)
                    
                if len(cut_points) == 1:
                    cut_points = [-np.inf] + cut_points + [np.inf]
                df = pd.DataFrame({'x': x, 
                                   'y': y, 
                                   'bucket': pd.cut(x, cut_points, include_lowest=True)})
                WOE_dic, IV = self.woe_single_x(df.bucket, df.y, event=1)
                return WOE_dic, IV
            else:
                df = pd.DataFrame({'x' : x,
                                   'y' : y,
                                   'bucket' : x})
                WOE_dic_raw, IV_raw = self.woe_single_x(df.bucket, df.y, event=1)
                return WOE_dic_raw, IV_raw
                # df['woe'] = df.bucket.map(lambda x : WOE_dic_raw.get(x, -999))
                # cut_points = self.get_cut_points_by_tree(df['woe'], df['y'])
                # df['woe_bucket'] = pd.cut(df['woe'], cut_points, include_lowest=True)
                # WOE_dic, IV = self.woe_single_x(df['woe_bucket'], df['y'], event=1)
                # return WOE_dic, IV
        except:
            # print(traceback.format_exc())
            print('__ERROR__')
            return np.nan, np.nan
            
