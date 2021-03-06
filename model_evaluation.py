# -*- coding: utf-8 -*-
"""
utils function for model evaluation

Created on Tue Mar 13 10:04:57 2018

@author: Changxun Fan
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib as mpl   # mpl.style.available
mpl.style.use('ggplot')

__ALPHA__ = 0.6
__LW__ = 2

def cal_metrics(y_true, y_pred, thr_point=None, precision=3):
    '''
    thr_point: 阈值
    画出该模型的AUC图，并返回KS，AC，PR，RC四个值
    '''
    fpr, tpr, thr = metrics.roc_curve(y_true, y_pred)

    ks_list = tpr - fpr
    ks = round(max(ks_list), precision)
    # 计算阈值
    if not thr_point:
        idx = np.argmax(tpr - fpr)
        thr_point = thr[idx]

    y_pred_2 = pd.Series(y_pred).map(lambda x: 0 if x < thr_point else 1) 
    ac = round(metrics.auc(fpr, tpr), precision)
    pr = round(sum(y_pred_2 == 0) / len(y_true), precision)
    rc = round(sum((y_pred_2 == 0) & (y_true == 1)) / sum(y_pred_2 == 0), precision)
        
    return ks, ac, pr, rc, fpr, tpr, thr_point


def plot_ks(y_true, y_pred, text='', ax=None, dpi=100):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)

    df = pd.DataFrame()
    df['y_true'] = np.array(y_true)
    df['y_pred'] = np.array(y_pred)
    df = df.sort_values('y_pred', ascending=False)

    neg, pos, thr = metrics.roc_curve(df.y_true, df.y_pred)
    ks = pos - neg
    t, s = sorted(y_pred), len(y_pred)
    # prob = [sum((df['y_pred'] >= i)) / df.shape[0] for i in thr]
    prob = [0] + [(s - t.index(i)) / s for i in thr[1:]]

    threshold = prob[np.argmax(np.array(pos) - np.array(neg))]
    # max_ks = ks_2samp(df['y_pred'][df['y_true'] == 1], df['y_pred'][df['y_true'] == 0])[0]
    max_ks = np.max(np.array(pos) - np.array(neg))
    best_thr = thr[np.argmax(np.array(pos) - np.array(neg))]

    ax.plot(prob, pos, lw=__LW__, label='TPR')
    ax.plot(prob, neg, lw=__LW__, label='FPR')
    ax.plot(prob, ks,  lw=__LW__, label='KS (%0.2f)' % max_ks)
    # ax.plot([threshold, threshold], [0, 1], lw=__LW__, linestyle='--',
    #          label=' best passing rate (%0.0f%%)\n(best threshold = %0.2f)' % (threshold * 100, best_thr))
    ax.set_xlim([-0.03, 1.03])
    ax.set_ylim([-0.03, 1.03])
    ax.set_title(f'{text} K-S curve')
    ax.legend(loc='best')
    return threshold, max_ks


def plot_ks_2(y_true, y_pred, text='', ax=None, dpi=100):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks = max(tpr - fpr)
    x = [1.0 * i / len(tpr) for i in range(len(tpr))]
    
    cut_index = (tpr - fpr).argmax()
    cut_x = 1.0 * cut_index / len(tpr)
    cut_tpr = tpr[cut_index]
    cut_fpr = fpr[cut_index]
    # plt.rcdefaults()#重置rc所有参数，初始化
    # plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    # plt.rc('figure', figsize=(8,6))
    # plt.figure(figsize = (8, 6), dpi=100) 
    ax.plot(x, tpr, lw=2, label='TPR')
    ax.plot(x, fpr, lw=2, label='FPR')    
    ax.plot([cut_x, cut_x], [cut_tpr, cut_fpr], color='firebrick', ls='--')  
    ax.text(0.45, 0.3, 'KS = %0.2f' % ks)       
    ax.set_xlabel('Proportion')
    ax.set_ylabel('Rate')
    ax.set_title(f'{text} K-S curve')
    ax.legend(loc='best')
    
           
def plot_lift(y_true, y_pred, text='', num_of_bins=100, ax=None, dpi=100):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)

    y_true, y_pred = pd.Series(y_true), pd.Series(y_pred)
    cut_points = [-np.inf] + get_cut_points_by_freq(-y_pred, num_of_bins=num_of_bins)[1: -1] + [np.inf]
    df_bin = bins_points(-y_pred, cut_points, labels=list(range(1, len(cut_points))))
    df = pd.DataFrame({'y_true': list(y_true), 'y_pred': list(y_pred), 'level': list(df_bin)})
    df['acc_count'] = 1

    df_lift = df.groupby('level')[['acc_count', 'y_true']].sum().cumsum()
    df_lift['acc_mean'] = df_lift['y_true'] / df_lift['acc_count']
    df_lift['lift'] = df_lift['acc_mean'] / np.mean(y_true)

    ax.plot(df_lift.index.tolist(), df_lift['lift'].tolist(), lw=2, label=text)
    for i in [1, 5, 10]:
        try:
            cut_x = int(i * (1 / num_of_bins) * 100)
            cut_y = df_lift.iloc[cut_x - 1]['lift']  
            ax.plot([cut_x], [cut_y], 'o', label=f'Top {cut_x} pct Lift:{np.round(cut_y, 1)}')
        except:
            pass
    ax.set_xlabel('Proportion')
    ax.set_ylabel('Lift')
    ax.set_title(f'{text} Lift Curve')
    ax.legend(loc='best')


def plot_evaluation(df_dic, path='model_evaluations.png', ascending=False, isolate=False, is_tick=False, dpi=100):
    """
    Args:
        df_dic:
        path:
        ascending: 
    
    Returns:
    
    Exampel:
        plot_evaluation({'YF_v2 + App (train)': [y_train, y_pred_all_train], 'YF_v2 + App (test)': [y_test, y_pred_all]})
    """
    num_col, num_row = len(df_dic), 4
    fig, axs = plt.subplots(num_row, num_col, figsize=(8 * num_col, 6 * num_row), dpi=dpi)
    
    label_list = list(df_dic.keys())
    y_true_list = [df_dic[lab][0] for lab in label_list]
    y_pred_list = [df_dic[lab][1] for lab in label_list]
    
    sub_ax = axs[0][0] if num_col > 1 else axs[0]
    plot_auc(y_true_list, y_pred_list, label_list, ax=sub_ax)
    

    cp = [-np.inf] + get_cut_points_by_freq(y_pred_list[0])[1:-1] + [np.inf]
    if ascending:
        lb = list(range(len(cp) - 1, 0, -1))
    else:
        lb = list(range(1, len(cp)))

    cp = None if isolate else cp

    for i, label in enumerate(df_dic.keys()):
        y_true, y_pred = list(df_dic[label][0]), list(df_dic[label][1])
        
        # plot-ks
        sub_ax = axs[1][i] if num_col > 1 else axs[1]
        plot_ks_2(y_true, y_pred, label, ax=sub_ax)
        df_bins = sta_groups(y_true, y_pred, cut_points=cp, labels=lb)
        up1, up2 = df_bins['single_overdue_rate'].max() * 120, df_bins['count'].max() * 1.4
        
        # plot-lift
        sub_ax = axs[2][i] if num_col > 1 else axs[2]
        plot_lift(y_true, y_pred, label, ax=sub_ax)
        
        # sorting-ability
        sub_ax = axs[3][i] if num_col > 1 else axs[3]
        sorting_ability(df_bins, upper1=up1, upper2=up2, text=label, ax=sub_ax, is_tick=is_tick)
        
        # sub_ax = axs[4][i] if num_col > 1 else axs[4]
        # plot_acc_od_ps_rc(df_bins, text=label, ax=sub_ax)
        plt.tight_layout()
    plt.savefig(path)


def get_cut_points_by_freq(x, num_of_bins=10):
    interval = 100 / num_of_bins
    cp = sorted(set(np.percentile(x,  i * interval) for i in range(num_of_bins + 1)))
    return list(cp)


def bins_freq(data, num_of_bins=10, labels=None, precision=4):
    '''
    分箱 - 按照相同的频率分箱
    
    data:    list/pd.Series 数据，用于分箱，一般为分数的连续值
    num_of_bins: 箱子的个数
    labels:  个数和bins的个数相等md5
    
    return:  分箱后的label
    '''
    if labels == None:
        r = pd.qcut(data, q=np.linspace(0, 1, num_of_bins+1), precision=precision, retbins=True, duplicates='drop')
    else:
        r = pd.qcut(data, q=np.linspace(0, 1, num_of_bins+1), precision=precision, retbins=True, labels=labels, duplicates='drop')
    return r[0]


def bins_points(data, cut_points, labels=None):
    '''
    分箱 - 按照给定的几个cut points分箱

    labels:  个数比cut_points的个数少1
    cut_points: 必须倒掉递增

    return:  分箱后的label
    '''
    if float('inf') not in cut_points:
        # cut_points = [min(data)] + cut_points + [max(data)]
        cut_points = [-np.inf] + cut_points + [np.inf]

    if labels == None:
        r = pd.cut(data, bins=cut_points, include_lowest=True)
    else:
        r = pd.cut(data, bins=cut_points, labels=labels, include_lowest=True)
    return r


def sorting_ability(df, upper1=50, upper2=100, text='', is_tick=False, is_asce=False, ax=None, precision=3, dpi=100):
    from matplotlib.ticker import FuncFormatter 
    def to_percent(temp, position):  
        return '%1.0f'% temp + '%' 
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
    ax2 = ax.twinx()                                    # 创建第二个坐标轴（设置 xticks 的时候使用 ax）
    width = 0.5

    # 柱状图
    plt_x = df.y_pred_level.tolist()
    plt_x_range = [r.replace('-', '') for r in df.y_pred_range.tolist()]

    plt_y = df['count']
    ax2.bar(plt_x, plt_y, width, alpha=__ALPHA__)

    # 折线图
    # 单箱逾期率
    plt_y = round(df.single_overdue_rate, 3) * 100
    ax.plot(plt_x, plt_y, linestyle='--', lw=__LW__, 
             label='overdue rate', marker='o')
    for i, (a, b) in enumerate(zip(plt_x[1:], plt_y[1:])):
        ax.text(a, b + 1, f'{round(b, precision)}%', ha='center', va='bottom', fontsize=8) 

    # 累计逾期率
    plt_y = round(df.acc_overdue_rate, 3) * 100
    ax.plot(plt_x, plt_y, linestyle='--', lw=__LW__, 
             label='accumulate overdue rate', marker='o')
    for i, (a, b) in enumerate(zip(plt_x, plt_y)):
        ax.text(a, b - 2.5, f'{round(b, precision)}%', ha='center', va='bottom', fontsize=8)

    if is_asce:
        ax.set_xlabel('Groups(Bad -> Good)')
    else:
        ax.set_xlabel('Groups(Good -> Bad)')
    ax.set_ylabel('Percentage')
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax2.set_ylabel('The number of person')

    ax.set_ylim([-upper1 * 0.02, upper1])
    ax2.set_ylim([-upper2 * 0.02, upper2])
    plt.xticks(rotation=90)

    plt.xticks(plt_x)
    if is_tick:
        # plt_x_range = [r.replace('-', '') for r in plt_x_range]
        plt.sca(ax)
        plt.xticks(plt_x, plt_x_range)
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    ax.legend(loc='best')
    ax.set_title(f'{text} sorting ability')


def plot_acc_od_ps_rc(df, text='', precision=3, is_text=True, is_tick=False, is_asce=False, ax=None, dpi=100):
    #### 累积逾期率、通过率和好人召回率图
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)

    plt_x = df.y_pred_level.tolist()
    plt_x_range = df.y_pred_range.tolist()

    # 累积逾期率
    plt_y = round(df.acc_overdue_rate, 3) * 100
    ax.plot(plt_x, plt_y, 
             linestyle='--', lw=__LW__, label='accumulate overdue rate',
             marker='o') 
    if is_text:
        for i, (a, b) in enumerate(zip(plt_x, plt_y)):
            ax.text(a, b + 1, f'{round(b, precision)}%', ha='center', va='bottom', fontsize=8)

    # 通过率 
    plt_y = round(df.acc_pass_rate, precision) * 100
    ax.plot(plt_x, plt_y, 
             linestyle='--', lw=__LW__, label='accumulate passing rate',
             marker='o') 
    if is_text:
        for i, (a, b) in enumerate(zip(plt_x, plt_y)):
            ax.text(a + 0.2, b - 4, f'{round(b, precision)}%', ha='center', va='bottom', fontsize=8) 

    # 累积好人召回率
    plt_y = round(df.acc_recall_rate_good, precision) * 100
    ax.plot(plt_x, plt_y, 
             linestyle=':', lw=__LW__, label='accumulate good person recall rate',
             marker='o')
    
    if is_text:
        for i, (a, b) in enumerate(zip(plt_x, plt_y)):
            ax.text(a, b + 2, f'{round(b, precision)}%', ha='center', va='bottom', fontsize=8) 
    
    if is_asce:
        ax.set_xlabel('Groups(Bad -> Good)')
    else:
        ax.set_xlabel('Groups(Good -> Bad)')
        
    ax.set_xticks(plt_x)
    if is_tick:
        ax.set_xticks(plt_x, plt_x_range)
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    
    ax.set_ylim([-3, 105])
    ax.set_ylabel('Percentage')
    ax.set_title(f'{text} dataset recall-pass curve')
    ax.legend(loc='best')


def plot_pr(y_true, y_pred, text='', pos_label=1, ax=None, dpi=100):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
    p, r, _ = metrics.precision_recall_curve(y_true, y_pred, pos_label=pos_label)
    f1 = 2 * p * r / (p + r)

    ax.plot(r, p, lw=__LW__, label='P-R curve')
    ax.plot(r, f1,lw=__LW__, label='F1 curve')
    ax.set_xlim([-0.03, 1.03])
    ax.set_ylim([-0.03, 1.03])
    ax.set_xlabel('Recall Rate')
    ax.set_ylabel('Precision Rate')
    ax.set_title(f'{text} P-R curve')
    ax.legend(loc='best')
    
    
def plot_auc(y_true_list, y_pred_list, label_list, precision=3, ax=None, dpi=100):
    # 计算 阈值
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
    ks, ac, pr, rc, fpr, tpr, thr_point = cal_metrics(y_true_list[0], y_pred_list[0], precision=precision)

    ax.set_title('Receiver Operating Characteristic')
    ax.plot([0, 1], [0, 1], 'r--')

    for i in range(len(label_list)):
        y_true = y_true_list[i]
        y_pred = y_pred_list[i]
        label = label_list[i]
        ks, ac, pr, rc, fpr, tpr, _ = cal_metrics(y_true, y_pred, thr_point=thr_point, precision=precision)
        ax.plot(fpr,  tpr,   label=f'{label} AUC = {ac}')

    ax.legend(loc='best')
    ax.set_xlim([-0.03, 1.03])
    ax.set_ylim([-0.03, 1.03])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')


def stat_level(df, level, score, target):
    df2 = df[[level, score, target]]
    df2['tmp'] = 1
    df_tmp = df2.pivot_table(index=level, columns=['tmp'], 
                    aggfunc={score : [np.mean, min, max], target : [np.mean, len, sum]})
    df_tmp.reset_index(drop=False, inplace=True)
    df_tmp.columns = ['y_pred_level', 'count', 'single_overdue_rate', 'hit', 'score_max', 'score_mean', 'score_min']
    df_tmp['y_pred_range'] = df_tmp['score_min'].astype(str) + '~' + df_tmp['score_max'].astype(str)
    df_tmp['acc_count'] = df_tmp['count'].cumsum()
    df_tmp['acc_hit'] = df_tmp['hit'].cumsum()
    df_tmp['acc_overdue_rate'] = df_tmp['acc_hit'] / df_tmp['acc_count']
    return df_tmp
    
    
def sta_groups(y_true, y_pred, cut_points=None, labels=list(range(1, 11))):
    df = pd.DataFrame({'y_true' : y_true, 'y_pred' : y_pred})
    df['count'] = 1
    df['y_true_opp'] = df['y_true'].replace([0, 1], [1, 0])
    
    # bins 合并在一起
    if cut_points:
        df['y_pred_level'] = bins_points(df.y_pred, cut_points=cut_points, labels=labels).astype(int)
        df['y_pred_range'] = bins_points(df.y_pred, cut_points=cut_points).astype(str)
        # df['y_pred_range'] = pd.cut(df.y_pred, bins=cut_points, labels=labels, include_lowest=True).astype(int)
    else:
        df['y_pred_level'] = bins_freq(df.y_pred, num_of_bins=10, labels=labels).astype(int)
        df['y_pred_range'] = bins_freq(df.y_pred, num_of_bins=10).astype(str)
        # df['y_pred_range'] = pd.qcut(df.y_pred, q=bins, duplicates='drop')
        # df['y_pred_range'] = df['y_pred_range'].cat.remove_unused_categories().cat.codes + 1

    # 需要改
    tmp = df.groupby(['y_pred_level', 'y_pred_range'])

    # 单箱逾期率
    sing_overdue = tmp.mean().reset_index()
    final_df = sing_overdue[['y_pred_level', 'y_pred_range', 'count', 'y_true']]
    final_df.columns = ['y_pred_level', 'y_pred_range', 'count', 'single_overdue_rate']
    final_df['count'] = tmp.sum()['count'].tolist()
    
    # 累计逾期率
    acc_overdue = tmp.sum().cumsum().reset_index()
    final_df['acc_overdue_rate'] = acc_overdue['y_true'] / acc_overdue['count']

    # 累计好人和坏人召回率
    sum_good_person = df['y_true_opp'].sum()
    sum_bad_person = df['y_true'].sum()
    sum_all_person = len(df)
    final_df['acc_recall_rate_good'] = acc_overdue['y_true_opp'] / sum_good_person
    final_df['acc_recall_rate_bad'] = acc_overdue['y_true'] / sum_bad_person
    final_df['acc_pass_rate'] = acc_overdue['count'] / sum_all_person
    return final_df


def model_cost_cmpt(df, clf, dic_model, label):
    """ 成本核算
    df: 数据
    clf: 模型
    dic_model: 特征组合列表
    label: 目标
    """
    model_result = {}
    for k, v in dic_model.items():
        # print(k)
        cols = sorted(v)

        tmp_X = df[cols]
        tmp_y = df[label]
        
        x_train, x_test, y_train, y_test = train_test_split(tmp_X, tmp_y, test_size=0.2, random_state=2017, stratify=tmp_y)

        ac = clf.train(x_train, y_train, x_test, y_test)
        y_pred_test = clf.predict(x_test)

        model_result[k] = [list(y_test), list(y_pred_test)]
        print()

    return model_result



def plot_line(x, y, text, xlabel='', ylabel='', ax=None, dpi=100):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
    x, y = pd.Series(x), pd.Series(y)
    ax.plot(x, y, lw=2, label=text, marker='o')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')


def plot_overdue(dic_bins, isolate=True, dpi=100):
    if not isolate:
        fig, axs = plt.subplots(1, 2, figsize=(8 * 2, 6 * 1), dpi=dpi)
        for k, bins in dic_bins.items():
            x = bins['y_pred_level']
            y1 = bins['single_overdue_rate']
            y2 = bins['acc_overdue_rate']
            plot_line(x, y1, f'{k} single overdue rate',
                      xlabel='level', ylabel='single-overdue', ax=axs[0])
            plot_line(x, y2, f'{k} acc overdue rate',
                      xlabel='level', ylabel='acc-overdue', ax=axs[1])
        plt.tight_layout()
        axs[0].set_title('single_overdue_rate')
        axs[1].set_title('acc_overdue_rate')
    plt.show()

def model_compare(df, model_list, pid_list, target='1d', score='score', cut_points=None):
    dic = {}
    dic_bin = {}
    ModelMonitor = umm.ModelMonitor()
    for i in model_list:
        y = df[(df.model_record_model_id == i) & (
            df.product_deadline.isin(pid_list))][target]
        s = df[(df.model_record_model_id == i) & (
            df.product_deadline.isin(pid_list))][score]

        if cut_points is None:
            cut_points = [-np.inf] + \
                ModelMonitor.get_cut_points_by_freq(s)[1:-1] + [np.inf]
        lb = list(range(len(cut_points) - 1, 0, -1))
        df_bins = sta_groups(y, s, cut_points=cut_points)
        # df_bins = sta_groups(y, s, cut_points=cut_points, labels=lb)
        dic_bin['M' + str(i)] = df_bins
        dic['M' + str(i)] = [y, -s]
    me.plot_evaluation(dic, isolate=True)
    plot_overdue(dic_bin, isolate=False)
    return dic_bin
