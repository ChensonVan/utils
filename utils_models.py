# coding: utf-8
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from scipy.interpolate import spline  

import xgboost as xgb

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


params_tree = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',  # 多分类的问题
    # 'num_class': 2,                  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,                    # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 6,                  # 构建树的深度，越大越容易过拟合
    'lambda': 2,                     # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,                # 随机采样训练样本
    'colsample_bytree': 0.7,         # 生成树时进行的列采样
    'scale_pos_weight': 1,
    'silent': 1,                     # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.1,                      # 学习率
    'seed': 1000,
    # 'nthread': 7,                    # cpu 线程数
    'eval_metric': 'auc'
}

params_linear = {
    'booster': 'gblinear',
    'objective': 'binary:logistic',  # 多分类的问题
    # 'num_class': 2,                  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,                    # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 6,                  # 构建树的深度，越大越容易过拟合
    'lambda': 2,                     # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,                # 随机采样训练样本
    'colsample_bytree': 0.7,         # 生成树时进行的列采样
    'scale_pos_weight': 1,
    'silent': 1,                     # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.1,                      # 学习率
    'seed': 1000,
    # 'nthread': 7,                    # cpu 线程数
    'eval_metric': 'auc'
}


def cal_overdue(df, tag='datediff', od_value=[0, 1, 3, 7, 14, 30]):
    '''
    统计逾期率
    '''
    if tag not in df.columns.tolist():
        return
    for i in od_value:
        df[f'{i}d'] = df[tag].apply(lambda x: 0 if x > -1000 and x < (i+1) else 1)
    del df[tag]


###################################################################
# 画图
###################################################################
def cal_metrics(clf, x, y, thr_point=None, pos_label=1):
    '''
    clf:
    x, y: 分别为
    thr_point: 阈值
    画出该模型的AUC图，并返回KS，AC，PR，RC四个值
    '''
    pred_y = clf.predict(x)
    fpr, tpr, thr = metrics.roc_curve(y, pred_y, pos_label=pos_label)

    ks_list = tpr - fpr
    ks = round(max(ks_list), 2)
    # 计算阈值
    if not thr_point:
        idx = np.argmax(tpr - fpr)
        thr_point = thr[idx]
    pred_y_2 = pd.Series(pred_y).map(lambda x: 0 if x < thr_point else 1) 
    ac = round(metrics.auc(fpr, tpr), 2)
    pr = round(sum(pred_y_2 == 0) / len(y), 2)
    rc = round(sum((pred_y_2 == 0) & (y == 1)) / sum(pred_y_2 == 0), 2)
        
    return ks, ac, pr, rc, fpr, tpr, thr_point

###################################################################
# K-S图
###################################################################
from scipy.stats import ks_2samp
def ks_curve(y_true, y_pred, bins=1000, pos_label=1):
    ascending = False if pos_label else True
    df = pd.DataFrame()
    df['y_true'] = np.array(y_true)
    df['y_pred'] = np.array(y_pred)
    df = df.sort_values('y_pred', ascending=ascending)

    if df.shape[0] > bins:
        step = int(df.shape[0] / bins)
    else:
        step = 1

    pos_num = df['y_true'].sum()
    neg_num = df.shape[0] - pos_num
    prob = []
    pos = []
    neg = []
    ks = []

    for i in range(0, df.shape[0], step):
        temp = df.iloc[:i, :]
        p_l = temp[(temp['y_true'] == 1)].shape[0]
        n_l = temp[(temp['y_true'] == 0)].shape[0]
        pos.append(p_l / pos_num)
        neg.append(n_l / neg_num)
        prob.append(i / df.shape[0])
        ks.append(pos[-1] - neg[-1])

    if i < df.shape[0]:
        i = df.shape[0]
        temp = df.iloc[:i, :]
        p_l = temp[(temp['y_true'] == 1)].shape[0]
        n_l = temp[(temp['y_true'] == 0)].shape[0]
        pos.append(p_l / pos_num)
        neg.append(n_l / neg_num)
        prob.append(i / df.shape[0])
        ks.append(pos[-1] - neg[-1])

    threshold = prob[np.argmax(np.array(pos) - np.array(neg))]
    #     max_ks = np.max(np.array(pos) - np.array(neg))
    max_ks = ks_2samp(df['y_pred'][df['y_true'] == 1], df['y_pred'][df['y_true'] == 0])[0]

    lw = 2
    plt.figure(figsize = (8, 6), dpi=100) 
    plt.plot(prob, pos, color='darkorange',
            lw=lw, label='TPR')
    plt.plot(prob, neg, color='darkblue',
            lw=lw, label='FPR')
    plt.plot(prob, ks, color='darkred',
            lw=lw, label='KS (%0.2f)' % max_ks)
    plt.plot([threshold, threshold], [0, 1], color='lightgreen', lw=lw, linestyle='--',
            label='threshold (%0.2f)' % threshold)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('K-S curve')
    plt.legend(loc="upper left")
    plt.show()
    return threshold, max_ks


def __model_evaluation(clf, x_tra, x_val, x_test, y_tra, y_val, y_test, bins=1000):
    '''
    clf: 训练的模型
    x_tra, x_val, x_test: 分别为 训练数据，验证数据，和最终的测试数据集
    y_tra, y_val, y_test:
    '''
    # fig, ax = plt.subplots(figsize=(10, 8))
    _ = clf.train(x_tra, y_tra, x_val, y_val)
    ks_tra,  auc_tra,  pr_tra,  rc_tra,  fpr_tra,  tpr_tra,  thr_point = cal_metrics(clf, x_tra, y_tra)
    ks_val,  auc_val,  pr_val,  rc_val,  fpr_val,  tpr_val,  _ = cal_metrics(clf, x_val, y_val, thr_point)
    ks_test, auc_test, pr_test, rc_test, fpr_test, tpr_test, _ = cal_metrics(clf, x_test, y_test, thr_point)

    # 画图部分 - AUC
    plt.figure(figsize = (8, 6), dpi=100) 
    plt.title('Receiver Operating Characteristic')
    plt.grid(True)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr_tra,  tpr_tra,  label=f'train AUC = {auc_tra}')
    plt.plot(fpr_val,  tpr_val,  label=f'valid AUC = {auc_val}')
    plt.plot(fpr_test, tpr_test, label=f'test  AUC = {auc_test}')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # K-S图
    y_pred = clf.predict(x_tra)
    th_tra, max_ks_tra = ks_curve(y_tra, y_pred, bins=bins)

    y_pred = clf.predict(x_val)
    th_val, max_ks_val = ks_curve(y_val, y_pred, bins=bins)

    y_pred = clf.predict(x_test)
    th_test, max_ks_test = ks_curve(y_test, y_pred, bins=bins)

    res = pd.DataFrame([[auc_tra,auc_val,auc_test],
                        [ks_tra, ks_val, ks_test],
                        [max_ks_tra, max_ks_val, max_ks_test],
                        [th_tra, th_val, th_test],
                        [pr_tra, pr_val, pr_test], 
                        [rc_tra, rc_val, rc_test]], 
                        columns=['train','val','test'],
                        index=['auc', 'ks', 'max_ks', 'threshold', 'pr', 'rc']).T
    return round(res, 3)



from sklearn.model_selection import train_test_split
def model_evaluation(clf, x, y, x_test, y_test, test_size=0.2, random_state=2017):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return __model_evaluation(clf, x_train, x_val, x_test, y_train, y_val, y_test)



def check_cv(clf, x, y, x_test, y_test, cv=5, random_state=1, sclient=0):
    '''
    x, y, x_test, y_test  分别为测试集和最终的测试集数据
    只支持 自己传入模型测试cv
    '''
    print('size =', x.shape)
    res_list = []
    aucs = []
    cv = StratifiedKFold(n_splits=cv, random_state=random_state) 
    for train_index, val_index in cv.split(x, y): 
        x_train, x_val = x.iloc[train_index,], x.iloc[val_index] 
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        res = __model_evaluation(clf, x_train, x_val, x_test, y_train, y_val, y_test)
        res_list.append(res)
        # model, auc = clf.train(x_tra, y_tra, x_val, y_val)
    return res_list



        
def cal_fi(clf, plot_flag=False):
    try:
        feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
    except:
        feat_imp = pd.Series(clf.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp = pd.DataFrame(feat_imp)
    feat_imp.reset_index(inplace=True)
    feat_imp.columns = ['feature', 'importance']

    if not plot_flag:
        return feat_imp

    x_pos = list(range(len(feat_imp)))
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap1 = cm.ScalarMappable(colors.Normalize(min(feat_imp.importance), max(feat_imp.importance), cm.hot))
    plt.bar(x_pos, feat_imp.importance, align='center', alpha=0.7, width=0.5, color=cmap1.to_rgba(feat_imp.importance))
    plt.xticks(x_pos, feat_imp.feature)
    plt.ylabel('Score')
    plt.title('Feature Importance Score')
    plt.setp(plt.gca().get_xticklabels(), rotation=70)
    # 标数字
    for a, b in zip(x_pos, feat_imp.importance):
        plt.text(a, b + 2, f'{b}', ha='center', va='bottom', fontsize=14)
    plt.show()
    return feat_imp
            
        


# XGB 参数调优
# 需要改动
import matplotlib.colors as colors
import matplotlib.cm as cm
def model_fit(clf, x, y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    '''
    clf 是xgboost的sklearn包的模型
    '''
    # 使用CV找到最佳的 n_estimators
    if useTrainCV:
        xgb_param = clf.get_xgb_params()
        xgtrain = xgb.DMatrix(x, label=y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        clf.set_params(n_estimators=cvresult.shape[0])

    print('clf.n_estimators =', clf.n_estimators)
     
    dmtrix = xgb.DMatrix(x, label=y)    
    
    # Fit the clforithm on the data
    clf.fit(x, y, eval_metric='auc')

    # Predict training set:
    pred_y = clf.predict(x)
    prob_y = clf.predict_proba(x)[:, 1]

    # Print model report:
    accuracy  = round(metrics.accuracy_score(y, pred_y), 2)
    auc_score = round(metrics.roc_auc_score(y, prob_y), 2)
    print('\nModel Report')
    print('Accuracy : {accuracy}')
    print('AUC Score (Train): {auc_score}')

    feat_imp = cal_fi(clf, raw_api=False)
    
    # 模型的AUC图
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, thr = metrics.roc_curve(y, prob_y)
    plt.title('Receiver Operating Characteristic')
 
    plt.plot(fpr, tpr, label=f'AUC = {auc_score}')
    plt.legend(loc='best')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    

def find_best_params(x, y, clf, params, score_metric='roc_auc'):
    '''
    根据x y 和model，找到最佳的参数，并返回最佳的model
    clf 是xgboost的sklearn包的模型
    '''
    gsearch = GridSearchCV(
                estimator=clf, 
                param_grid=params, 
                scoring=score_metric,
                n_jobs=4,
                iid=False, 
                cv=5)
    gsearch.fit(x, y)
    return gsearch.best_estimator_


if __name__ == '__main__':
    raw = pd.read_csv('../model90/xuting/20180106_样本.csv')
    data = raw[['xinyan_score', 'baidu_score', 'kexin_score', 'yuqi_day', 'type']]
    data.shape


    od_value = [0, 1, 3, 7, 14]
    base = ['xinyan_score', 'baidu_score', 'kexin_score']
    cal_overdue(data, tag='yuqi_day', od_value=[0, 1, 3, 7, 14])

    data.head()
    data.shape


    data_train = data[data.type != 3]
    data_test = data[data.type != 3]

    x = data_train[base]
    y = data_train['d7']

    x_test = data_test[base]
    y_test = data_test['d7']

    size = 40000
    x_tmp = x.iloc[:size, :]
    y_tmp = y[:size]

    x_tmp.shape

    res = check_cv(x_tmp, y_tmp, x_test, y_test)
    x.fillna(0, inplace=True)



    # 1. 确定学习速率和tree_based 参数调优的估计器数目
    # from xgboost import XGBClassifier
    estimator = xgb.XGBClassifier(
             learning_rate=0.1,
             n_estimators=200,
             max_depth=6,
             min_child_weight=2,
             gamma=0,
             subsample=0.8,
             colsample_bytree=0.8,
             objective='binary:logistic',
             nthread=4,
             scale_pos_weight=1,
             seed=27)
    estimator

    model_fit(estimator, x, y)
    estimator

    estimator.feature_importances_

    feat_imp = pd.Series(estimator.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.importance

    # 2. max_depth 和 min_weight参数调优
    param_test2 = {
                    'max_depth' : list(range(3, 10, 1)),
                    'min_child_weight' : list(range(1, 6, 2))
    }

    estimator = find_best_params(x, y, estimator, param_test2)
    estimator

    # 3. gamma参数调优
    param_test3 = {
                    'gamma' : [i / 10.0 for i in range(0,5)]
    }

    estimator = find_best_params(x, y, estimator, param_test3)
    estimator

    # 4. 调整subsample 和 colsample_bytree 参数
    # 取0.6,0.7,0.8,0.9作为起始值
    param_test4 = {
                    'subsample' : [i / 10.0 for i in range(6,10)],
                    'colsample_bytree' : [i / 10.0 for i in range(6,10)]
    }

    estimator = find_best_params(x, y, estimator, param_test4)
    estimator

    # 5. 正则化参数调优
    # L1 : alpha
    # L2 : lambda
    param_test5 = {
                    'reg_alpha' : [1e-5, 1e-2, 0.1, 1, 100]
    }

    estimator = find_best_params(x, y, estimator, param_test5)
    estimator


    x.fillna(0)
    model_fit(estimator, x, y)

    estimator

    estimator.feature_importances_




