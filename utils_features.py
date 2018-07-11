# coding: utf-8
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib as mpl
# import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')

import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
import math

import os

def cal_overdue(df, col='overdue_days', targets=[0, 1, 3, 7, 14, 30], drop=True):
    '''
    统计逾期率
    '''
    if col not in df.columns.tolist():
        return
    for i in targets:
        df[f'{i}d'] = df[col].apply(lambda x: 0 if x > -1000 and x < (i+1) else 1)
    if drop:
        del df[col]


def sta_od_rate(df, targets=[0, 1, 3, 7, 14, 30]):
    '''
    统计各个天数的逾期率
    '''
    od = pd.DataFrame()
    dn = [f'{n}d' for n in targets]
    col = set(df.columns) - set(dn)
    for tag in col:
        tmp_df = df[df[tag].notnull()]
        for i in targets:
            od.loc[tag, f'{i}d'] = 1 - tmp_df[f'{i}d'].value_counts(normalize=True)[0]
    for i in targets:
        od.loc['ALL', f'{i}d'] = 1 - df[f'{i}d'].value_counts(normalize=True)[0]
    return np.round(od, 3)

def sta_cov_rate(df):
    '''
    测试集中各个指标的覆盖率
    '''
    cr = pd.DataFrame(1 - (df.isnull().sum() / len(df)), columns=['covRate']).T
    cr['ALL'] = len(df.dropna(how='any')) / len(df)
    return np.round(cr, 3)

def get_labels(df_meta, col, labels=[0, 1, 3, 7, 14, 30]):
    import datetime
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    df = df_meta.copy()
    od_col = 'overdue_days'
    if len(col) == 1:
        od_col = col[0]
    elif len(col) == 2:
        df[col[0]].replace('1000-01-01 00:00:00', nowTime, regex=True, inplace=True)
        df[od_col] = (pd.to_datetime(df[col[0]]) - pd.to_datetime(df[col[1]])).dt.days
    else:
        raise ValueError

    for i in labels:
        df[f'{i}d'] = df[od_col].apply(lambda x: 0 if x <= i else 1)
    return df


def bins_freq(data, num_of_bins=10, labels=None):
    '''
    分箱 - 按照相同的频率分箱

    data:    list/pd.Series 数据，用于分箱，一般为分数的连续值
    num_of_bins: 箱子的个数
    labels:  个数和bins的个数相等

    return:  分箱后的label
    '''
    # r = pd.qcut(data, q=np.linspace(0, 1, num_of_bins+1), precision=0, retbins=True, labels=labels)
    if labels == None:
        r = pd.qcut(data, q=np.linspace(0, 1, num_of_bins+1), precision=0, retbins=True)
    else:
        r = pd.qcut(data, q=np.linspace(0, 1, num_of_bins+1), precision=0, retbins=True, labels=labels)
        if isinstance(labels[0], int):
            return r[0].astype(int)
        elif isinstance(labels[0], float):
            return r[0].astype(float)
        elif isinstance(labels[0], str):
            if labels[0].isdigit():
                return r[0].astype(float)
    return r[0]


def bins_points(data, cut_points, labels=None):
    '''
    分箱 - 按照给定的几个cut points分箱

    labels:  个数比cut_points的个数少1
    cut_points: 必须倒掉递增

    return:  分箱后的label
    '''
    if float('inf') not in cut_points:
        cut_points = [min(data)] + cut_points + [max(data)]

    if labels == None:
        r = pd.cut(data, bins=cut_points, include_lowest=True)
    else:
        r = pd.cut(data, bins=cut_points, labels=labels, include_lowest=True)
        if isinstance(labels[0], int):
            return list(map(lambda x : int(x), r))
        elif isinstance(labels[0], float):
            return list(map(lambda x : float(x), r))
        elif isinstance(labels[0], str):
            if labels[0].isdigit():
                return list(map(lambda x : float(x), r))
    return r


def get_cut_points(x, y, max_depth=5, min_samples_leaf=0.01, max_leaf_nodes=None, random_state=7):
    '''
    根据决策树选出cut_points
    '''
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                max_leaf_nodes=max_leaf_nodes, random_state=random_state)
    dt.fit(np.array(x).reshape(-1, 1), np.array(y))
    th = dt.tree_.threshold
    f = dt.tree_.feature
    return sorted(th[np.where(f != -2)])


def bins_tree(x, cut_points, bins=20, interval=True):
    '''
    对feature数据离散化
    x: 连续性数值
    cut_points: list of cut_points, 根据决策树生成的
    bins:
    interval:
    '''
    if interval:
        if len(set(x)) > bins:
            cut_points.append(np.inf)
            cut_points.insert(0, -np.inf)
            x_cut = pd.cut(x, bins=cut_points)
            return x_cut
        else:
            return x
    else:
        x = np.array(x)
        y = x.copy()

        if len(set(x)) > bins:
            cut_points.append(np.inf)
            cut_points.insert(0, -np.inf)
            for i in range(len(cut_points) - 1):
                y[np.where((x > cut_points[i]) & (x <= cut_points[i + 1]))] = i + 1
            return y
        else:
            return x


def count_binary(a, event=1):
    '''
    统计0，1的个数
    '''
    event_count = (a == event).sum()
    non_event_count = a.shape[-1] - event_count
    return event_count, non_event_count


def cal_woe_iv(x, y, event=1, max_depth=5, min_samples_leaf=0.01, max_leaf_nodes=None, bins=20, interval=True,
           random_state=7):
    '''
    计算WOE
    x: 1-D 单个feature的数据
    y: 1-D target
    '''
    x = np.array(x)
    y = np.array(y)

    cut_points = get_cut_points(x, y, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,
                           random_state=random_state)
    x = bins_tree(x, cut_points, bins=bins, interval=interval)

    event_total, non_event_total = count_binary(y, event=event)
    x_labels = np.unique(x)
    woe_dict = {}
    iv = 0
    for x1 in x_labels:
        # this for array, 如果传入的是pd.Series，y会按照index切片，但是np.where输出的却是自然顺序，会出错
        y1 = y[np.where(x == x1)[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        rate_event = 1.0 * event_count / event_total
        rate_non_event = 1.0 * non_event_count / non_event_total
        if rate_event == 0:
            woe1 = -20
        elif rate_non_event == 0:
            woe1 = 20
        else:
            woe1 = math.log(rate_event / rate_non_event)
        woe_dict[x1] = woe1
        iv += (rate_event - rate_non_event) * woe1
    return woe_dict, iv


def iv_df(df, targets, columns=None, event=1, max_depth=5, min_samples_leaf=0.01,
          max_leaf_nodes=None, bins=20, interval=True, random_state=7):
    '''
    计算所有给定features的WOE，并计算IV
    df:       dataframe
    targets: 需要计算的lables的list，即targets
    columns:  如果给出，则算制定的columns，否则算除了od的所有的IV
    '''
    dic = defaultdict(dict)

    if columns is None:
        columns = [c for c in df.columns if c not in targets]

    for t in targets:
        t = f'{t}d'
        for c in columns:
            dic[t][c] = cal_woe_iv(df[c], df[t], event=event, max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,
                            bins=bins, interval=interval, random_state=random_state)[1]
    df = pd.DataFrame(dic)
    df.columns = [['IV'] * df.shape[1], df.columns]
    return df


import matplotlib.pyplot as plt
def cal_auc_ks_iv(df, targets=[0, 1, 3, 7, 14, 30], text='', max_depth=2, plot=True, precision=3):
    '''
    计算 AUC KS 和 IV的值
    并画出对应的AUC图
    '''
    ks = pd.DataFrame()
    ac = pd.DataFrame()
    iv = pd.DataFrame()

    dn = [f'{n}d' for n in targets]
    cols = set(df.columns) - set(dn)

    for n in targets:
        auc_value = []
        ks_value = []
        iv_value = []

        plt.figure(figsize=(6,4), dpi=100)
        for var in cols:
            y_true = df[df[var].notnull()][f'{n}d']
            y_pred = df[df[var].notnull()][var]

            # 计算各个指标的 fpr tpr 和 thr
            fpr, tpr, thr = roc_curve(y_true, y_pred, pos_label=1)

            # 计算AUC值
            ac_single = auc(fpr, tpr)
            if ac_single < 0.5:
                fpr, tpr, thr = roc_curve(y_true, -y_pred, pos_label=1)
                ac_single = auc(fpr, tpr)
            auc_value.append(ac_single)

            # 计算K-S值
            ks_single = (tpr - fpr).max()
            ks_value.append(ks_single)

            # 计算IV值
            iv_single = cal_woe_iv(y_pred, y_true, max_depth=max_depth)[1]
            iv_value.append(iv_single)

            if plot:
                # ROC Cureve
                plt.plot(fpr, tpr, lw=1, label=f'{var}(auc=' + str(round(ac_single, precision)) + ')')
                plt.plot(fpr, tpr, lw=1)

                # Labels
                plt.grid()
                plt.plot([0,1], [0,1], linestyle='--', color=(0.6, 0.6, 0.6))
                plt.plot([0, 0, 1], [0, 1, 1], lw=1, linestyle=':', color='black')
                plt.xlabel('false positive rate')
                plt.ylabel('true positive rate')
                plt.title(f'{text}ROC for {n}d')
                plt.legend(loc='best')

        auc_part = pd.DataFrame(auc_value, columns=[f'{n}d'], index=cols)
        ac = pd.concat([ac, auc_part], axis=1)

        ks_part  = pd.DataFrame(ks_value, columns=[f'{n}d'], index=cols)
        ks = pd.concat([ks, ks_part], axis=1)

        iv_part  = pd.DataFrame(iv_value, columns=[f'{n}d'], index=cols)
        iv = pd.concat([iv, iv_part], axis=1)

    iv = np.round(iv, precision)
    ac = np.round(ac, precision)
    ks = np.round(ks, precision)
    return ac, ks, iv


def cal_vif(df):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame(index=df.columns.tolist())
    vif_data = df
    vif['VIF'] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
    return np.round(vif.T, 3)


def cal_pearson_cor(df):
    return df.corr()


def sorting(df, targets=[0, 1, 3, 7, 14, 30], asc=[]):
    '''
    df:  待计算数据
    asc: 需要逆序的标签，返回值默认为从高到低排序
         分数越高越好的需要逆序排序，比如芝麻分，新颜分和氪信分
    '''
    data = df.copy()
    dn = [f'{n}d' for n in targets]
    col = set(data.columns) - set(dn)
    all_od = []

    tmp = defaultdict(pd.DataFrame)
    for s in col:
        x = df[s]
        # 连续值分箱， 需要无重复，等间隔？还是等频率？
        try:
            data[f'{s}_range'] = pd.qcut(x, q=np.linspace(0, 1, 11), precision=0, retbins=True)[0]
        except:
            data[f'{s}_range'] = pd.cut(x, bins=10, precision=0, retbins=True)[0]

        for i in targets:
            t = f'{i}d'
            # groupby 这里会少了
            d = data.groupby(f'{s}_range')[t].value_counts(normalize=True, sort=False).xs(1, level=t)
            d = d.to_frame(name = s + '_' + t)
            # axis=1 横向，indexs数量不符合, 理论上会填充NaN
            d.index = [str(_) for _ in d.index]
            tmp[f'{s}_od'].index = [str(_) for _ in tmp[f'{s}_od'].index ]
            tmp[f'{s}_od'] = pd.concat([tmp[f'{s}_od'], d], axis=1)

        tmp[f'{s}_od'].fillna(0, inplace=True)
        all_od += [tmp[f'{s}_od']]

    # 部分逆序
    tag = '_d' + str(targets[0])
    asc = [_ + tag for _ in asc]
    for i in range(len(all_od)):
        if all_od[i].columns.values[0] in asc:
            all_od[i].sort_index(ascending=True, inplace=True)
            continue
        all_od[i].sort_index(ascending=False, inplace=True)
    return all_od


def sorting_plot(df, all_od, targets=[0, 1, 3, 7, 14, 30], text=''):
    dn = [f'{n}d' for n in targets]
    col = set(df.columns) - set(dn)
    od = pd.DataFrame()

    for s, t in enumerate(targets):
        od_add = pd.DataFrame({dn[s]: [all_od[i].iloc[:,s] for i in range(len(all_od))]})
        od = pd.concat([od, od_add], axis=1)

        f = plt.figure(figsize=(7, 5), dpi=100)
        x = range(1, 11)
        for i, j in enumerate(col):
            plt.plot(x, od[f'{t}d'][i], linestyle='--', marker='o', ms=5, label=j)

        plt.grid(True)
        plt.xlabel('groups(bad -> good)');
        plt.ylabel('overdue rate')
        plt.title(f'{text} Sorting Ability for {t}d')
        plt.legend(loc='best')



def plot_overdue_with_bins(df_list, targets=[0, 1, 3, 7, 14, 30]):
    '''
    根据targets_list的值，绘制每一箱的逾期率
    '''
    bar_width = 0.5

    for df in df_list:
        fig, ax = plt.subplots(figsize=(16, 8))
        labels = list(df.index)
        bar_len = list(range(1, len(labels) + 1))
        tick_pos = [i for i in bar_len]

        tmp = np.zeros(len(labels))
        tmp2 = np.zeros(len(labels))

        for col in df.columns:
            overdue_dx = np.round(df[col].tolist(), 4)
            ax.bar(bar_len, overdue_dx, bottom=tmp, label=col, width=bar_width, alpha=0.7)
            tmp2 = [a+b/3 for a,b in zip(tmp, overdue_dx)]
            tmp = [sum(_) for _ in zip(tmp, overdue_dx)]
            # 折线图
            # ax.plot(bar_len, tmp, marker='o')

            # 标数字
            for x, y, z in zip(bar_len, tmp2, overdue_dx):
                z = round(z, 2)
                if round(z, 1) == 0.2:
                    plt.text(x, y, f'{z}', ha='center',  fontsize=10)
                elif round(z, 1) == 0.1:
                    plt.text(x, y, f'{z}', ha='center',  fontsize=8)
                else:
                    plt.text(x, y, f'{z}', ha='center',  fontsize=14)
        tag = ' '.join(col.split('_')[:-1])
        plt.title(f'{tag} Overdue Rate')
        plt.xticks(tick_pos, labels)
        plt.legend([f'{_}d' for _ in targets], loc='best')
        # plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='left')
        plt.grid()
        plt.show()


import matplotlib.colors as colors
import matplotlib.cm as cm
def plot_overdue_single(df, targets=[0, 1, 3, 7, 14, 30]):
    '''
    对columns里面的指标单一绘制，各个od的逾期率
    '''
    bar_width = 0.5
    x_pos = range(1, len(targets) + 1)
    labels = list(df.columns)
    bar_len = list(range(1, df.shape[1] + 1))
    tick_pos = [i for i in bar_len]

    for idx in df.index:
        fig, ax = plt.subplots(figsize=(10, 5))
        overdue_dx = np.round(df.loc[idx, :].tolist(), 4)
        cmap1 = cm.ScalarMappable(colors.Normalize(min(overdue_dx), max(overdue_dx), cm.hot))
        ax.bar(x_pos, overdue_dx, align='center', alpha=0.7, width=bar_width, color=cmap1.to_rgba(overdue_dx))
        ax.plot(x_pos, overdue_dx, marker='o')
        ax.set_ylabel("Percentage")
        ax.set_xlabel("")
        plt.xticks(tick_pos, labels)

        # 标数字
        for x, y in zip(x_pos, overdue_dx):
            plt.text(x, y+0.002, f'{y}', ha='center', va='bottom', fontsize=14)

        plt.legend([f'{_}d' for _ in targets], loc='best')
        plt.title(f'{idx} Overdue Rate')
        plt.setp(plt.gca().get_xticklabels(), rotation=0, horizontalalignment='left')
        plt.grid()
        plt.show()



#############################################
# 2018.04.04
#############################################
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf

def feature_selection(X, y, feature_n, method, alpha=0.05, stepwise=True):
    remining = set(X.columns.tolist())
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remining and best_new_score <= alpha and len(selected) < feature_n:
        scores_with_candidates = []
        for candidate in remining:
            x_candidates = selected + [candidate]
            try:
                if method == 'logistic':
                    model_stepwise_forward = sm.Logit(y, X[x_candidates]).fit(disp=False)
                elif method == 'linear':
                    model_stepwise_forward = sm.OLS(endog=y, exog=x[x_candidates]).fit(disp=False)
                else:
                    raise Exception('method must be in ["logistic", "linear"]')
            except:
                x_candidates.remove(candidate)
                print("\n\t feature " + candidate + " selection exception occurs")
                continue
            score = model_stepwise_forward.pvalues[candidate]
            scores_with_candidates.append((score, candidate))

        scores_with_candidates.sort(reverse=True)
        best_new_score, best_candidate = scores_with_candidates.pop()

        if best_new_score <= alpha:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            print(best_candidate +' enters: pvalue: ' + str(best_new_score))

        if stepwise:
            if method == 'logistic':
                model_stepwise_backford = sm.Logit(y, X[selected]).fit(disp=False)
            elif method == 'linear':
                model_stepwise_backford = sm.OLS(endog=y, exog=X[selected]).fit(disp=False)
            else:
                raise Exception('method must be in ["logistic", "linear"]')
            for i in selected:
                if model_stepwise_backford.pvalues[i] > alpha:
                    selected.remove(i)
                    print(i +' removed: pvalue: ' + str(model_stepwise_backford.pvalues[i]))
    return selected



if __name__ == '__main__':
    import pandas as pd
    df = pd.DataFrame({'datediff' : [1, 2, 3, 4, 5, 6, 7, 8, 9 , 10]})
    cal_overdue(df)
    print(df)
