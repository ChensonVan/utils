
import sys
import os
from .basic_model import BasicModel
from sklearn import metrics

import pandas as pd
import numpy  as np

sys.path.append('/Users/Chenson/Dropbox/KNJK/utils')


# scikit-learn（工程中用的相对较多的模型介绍）：1.11. Ensemble methods
# http://blog.csdn.net/mmc2015/article/details/47271195


#################################################################
# XGB
#################################################################
from xgboost.sklearn import XGBClassifier
class XGB_Classifier(BasicModel):
    def __init__(self):
        """ set parameters """
        self.model = None
        self.max_depth = 1                      # 构建树的深度，越大越容易过拟合
        self.learning_rate = 0.08
        self.n_estimators = 200
        self.silent = 0                         # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        self.objective = 'binary:logistic'
        self.booster = 'gbtree'
        self.nthread = 4                        # cpu 线程数 默认最大
        self.gamma = 0.5                        # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子
        self.min_child_weight = 30               # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
                                                # 假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
                                                # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        self.max_delta_step = 1                 # 最大增量步长，我们允许每个树的权重估计。
        self.subsample = 0.7                    # 随机采样训练样本 训练实例的子采样比
        self.colsample_bytree = 0.8             # 生成树时进行的列采样 
        self.colsample_bylevel = 0.8
        self.reg_alpha = 0.05                   # L1 正则项参数
        self.reg_lambda = 0.5                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        self.scale_pos_weight = 5               # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
        self.base_score = 0.5
        self.seed = 0
        # self.missing = float, optional
        self.metric = 'auc'

        self.num_rounds = 1000
        self.early_stopping_rounds = 15
        self.possible_label = 1


    def init_model(self, params):
        if params:
            self.model = XGBClassifier(params)
        else:
            self.model = XGBClassifier(
                            max_depth=self.max_depth,
                            learning_rate=self.learning_rate,
                            n_estimators=self.n_estimators,
                            silent=self.silent,
                            objective=self.objective,
                            # booster=self.booster,
                            nthread=self.nthread,
                            gamma=self.gamma,
                            min_child_weight=self.min_child_weight,
                            max_delta_step=self.max_delta_step,
                            subsample=self.subsample,
                            colsample_bytree=self.colsample_bytree,
                            colsample_bylevel=self.colsample_bylevel,
                            reg_alpha=self.reg_alpha,
                            reg_lambda=self.reg_lambda,
                            seed=self.seed
                            # base_score = 0.5,
                            ) 


#################################################################
# XGB原始接口
#################################################################
import xgboost as xgb
class XGB_Classifier2(BasicModel):
    def __init__(self):
        """ set parameters """
        self.model = None
        self.seed = 0
        self.num_rounds = 500
        self.params = {
                        'booster'       : 'gbtree',
                        'objective'     : 'binary:logistic',    # 多分类的问题
                        'num_class'     : 2,                    # 类别数，与 multisoftmax 并用
                        'gamma'         : 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
                        'max_depth'     : 6,                    # 构建树的深度，越大越容易过拟合
                        'lambda'        : 20,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                        'subsample'     : 0.7,                  # 随机采样训练样本
                        'colsample_bytree': 0.7,                # 生成树时进行的列采样
                        'scale_pos_weight': 1,
                        'silent'        : 1,                    # 设置成1则没有运行信息输出，最好是设置为0.
                        'eta'           : 0.01,                 # 如同学习率
                        'seed'          : 1000,
                        'nthread'       : 2,                    # cpu 线程数
                        'eval_metric'   : 'auc'
                    }


    def init_model(self):
        pass


    def train(self, x_train, y_train, x_val, y_val, params=None, is_eval=False, is_verbose=False):
        dtrain = xgb.DMatrix(x_train, y_train)
        dval = xgb.DMatrix(x_val, y_val)
        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        self.model = xgb.train(params, dtrain, self.num_rounds, watchlist, early_stopping_rounds=self.early_stopping_rounds)
        print('INFO : bset_iteration is', self.model.best_iteration)
        self.best_iteration = self.model.best_iteration

        y_pred = self.predict(x_val)
        auc_score = metrics.roc_auc_score(y_val,  y_pred)
        return auc_score


    def predict(self, x_test):
        """ return the predicted result of test data """
        if self.model == None:
            raise Exception('Please fit the data by using model before predict') 

        dtest = xgb.DMatrix(x_test)
        y_pred = self.model.predict(dtest)
        return y_pred


    def get_stacking(self, x, y, x_test, y_test, n_folds=5, random_state=2017):
        pass


    def cross_validation(self, x, y, x_test, y_test, n_folds=5, random_state=2017):
        from sklearn.model_selection import KFold, StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for i, (tra_idx, val_idx) in enumerate(SKF.split(x, y)):
            print(f'>>> {i} fold, train {len(tra_idx)}, val {len(val_idx)}')
            x_tra, y_tra = x[tra_idx], y[tra_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            

'''
#################################################################
# LGB
#################################################################
import lightgbm as lgb
class LGB_Classifier(BasicModel):
    def __init__(self):
        self.model = None
        self.num_boost_round = 2000
        self.early_stopping_rounds = 15
        self.metric = 'auc'
        self.objective = 'regression'
        self.n_estimators = 20
        self.learning_rate = 0.3
        self.num_leaves = 31
        self.min_sum_hessian_in_leaf = 1e-3
        self.min_gain_to_split = 10
        self.bagging_fraction = 0.8
        self.feature_fraction = 0.9
        self.lambda_l1 = 10
        self.lambda_l2 = 20
        self.possible_label = 1


    def init_model(self, params):
        if params:
            self.model = lgb.LGBMRegressor(params)
        else:
            self.model = lgb.LGBMRegressor(
                            objective=self.objective,
                            num_leaves=self.num_leaves,
                            learning_rate=self.learning_rate,
                            n_estimators=self.n_estimators,
                            min_sum_hessian_in_leaf=self.min_sum_hessian_in_leaf,
                            min_gain_to_split=self.min_gain_to_split,
                            bagging_fraction=self.bagging_fraction,
                            feature_fraction=self.feature_fraction,
                            lambda_l1=self.lambda_l1,
                            lambda_l2=self.lambda_l2
                            )        

'''

#################################################################
# RF
# http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#################################################################
from sklearn.ensemble import RandomForestClassifier 
class RF_Classifier(BasicModel):
    def __init__(self):
        self.model = None
        self.n_estimators = 20
        self.eval_metric = 'auc'
        self.early_stopping_rounds = 15


    def init_model(self, params):
        if params:
            self.model = RandomForestClassifier(params)
        else:
            self.model = RandomForestClassifier(
                            bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            max_depth=2, 
                            max_features='auto', 
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None,
                            min_samples_leaf=1, 
                            min_samples_split=2,
                            min_weight_fraction_leaf=0.0,
                            n_estimators=50, 
                            n_jobs=1,
                            oob_score=False, 
                            random_state=0, 
                            verbose=0, 
                            warm_start=False)        



#################################################################
# LR
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
#################################################################
from sklearn.linear_model import LogisticRegression
class LR_Classifier(BasicModel):
    def __init__(self):
        self.model = None
        self.eval_metric = 'RMSE'
        self.early_stopping_rounds = 15


    def init_model(self, params):
        if params:
            self.model = LogisticRegression(params)
        else:
            self.model = LogisticRegression()
