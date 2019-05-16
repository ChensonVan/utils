# coding: utf-8
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from collections import defaultdict
import math
import os, sys

sys.path.append('/data4/dm_share')
from mlearn import mlearn

from .utils_model_monitor import ModelMonitor
from .model_evaluation import cal_metrics


# 以下用于模型报告
def get_cover_ratio(df, time_col='apply_risk_created_at', time_format='%Y-%m-%d', 
                     time_span='day', decimals=4, exclude_cols=[]):
    df2 = df.notnull()
    feas = [fea for fea in df.columns if fea != time_col and fea not in exclude_cols]
    if time_span == 'day':
        df2['date'] = pd.to_datetime(df[time_col], format=time_format).dt.date
    elif time_span == 'month':
        df2['date'] = pd.to_datetime(df[time_col], format=time_format).astype(str).str[:7]
    else:
        raise ValueError("Error: Got unexpected time_span value.")
        
    tmp1 = df2.groupby('date')[[time_col]].agg(len).reset_index()
    tmp1.columns = ['date', 'count']
    tmp2 = df2.groupby('date')[feas].agg(np.mean).reset_index()
    return pd.merge(tmp1, tmp2, on='date')


def _get_col_widths(dataframe):
    # First we find the maximum length of the index column   
    # Then, we concatenate this to the max of the lengths of column name and its values for each column, left to right
    return [max([len(str(s)) for s in dataframe[col].values] + [len(col)]) for col in dataframe.columns]


def write_cov_ratio(df, file_name='conditional_format2.xlsx'):
    import xlsxwriter
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='cover_ratio', index=False)

    h, w = df.shape
    
    workbook  = writer.book
    worksheet = writer.sheets['cover_ratio']
        
    # idx = xlsxwriter.utility.xl_range_abs(1, 2, h, w-1).replace('$', '')
    idx = xlsxwriter.utility.xl_range_abs(1, 2, h, w-1)
    worksheet.conditional_format(idx, {'type': '3_color_scale'})

    sheet_format = workbook.add_format({'num_format': '0%'})
    worksheet.set_column(idx, None, sheet_format)
    
    for i, width in enumerate(_get_col_widths(df)):
        worksheet.set_column(i, i, width)
    writer.save()
    
    
def generate_data_report(clf, df_train, df_oot, fea_in, report_path, suffix, pic_path):
    import os
    if not os.path.exists(report_path):
        os.mkdir(report_path)
    report_dst = os.path.join(os.getcwd(), report_path)
        
    df = pd.concat([df_train, df_oot])
    df_cov_ratio = get_cover_ratio(df[fea_in], time_span='day')
    write_cov_ratio(df_cov_ratio, file_name=report_path  + f'/{suffix}_feature_cov_ratio.xlsx')
    
    mlearn.reporter.spliter_report(df_train[fea_in], df_oot[fea_in], 
                                   'apply_risk_created_at', '14d', report_dst)
    
    writer = pd.ExcelWriter(report_path + f'/{suffix}_Reports.xlsx', engine='xlsxwriter')
    df_report_woe = pd.read_pickle(report_path + '/woe_eva_report.report')
    df_report_stats = pd.read_pickle(report_path + '/data_stats.report')
    df_report_stable = pd.read_pickle(report_path + '/stable_test.report')
    
    # df_tmp3.to_excel(writer, sheet_name='Overview')
    df_report_stats.to_excel(writer, sheet_name='Model_Stats_Report')
    df_report_stable.to_excel(writer, sheet_name='Model_Stable_Report')
    df_report_woe.to_excel(writer, sheet_name='Model_WOE_Report')
    
    df_feature_importance1 = clf.get_fscore(importance_type='gain')
    df_feature_importance2 = clf.get_fscore(importance_type='weight')
    df_feature_importance1.to_excel(writer, sheet_name='Model_Feature_Importance_gain', index=False)
    df_feature_importance2.to_excel(writer, sheet_name='Model_Feature_Importance_weight', index=False)
    df_cov_ratio.to_excel(writer, sheet_name='Feature_Cov_Ratio', index=False)
    
    workbook = writer.book
    # workbook.filename = 'Reports_ALL.xlsm'
    # workbook.add_vba_project('../vbaProject.bin')

    worksheet = workbook.add_worksheet('Model_Evaluation')
    worksheet.set_column('B:B', 30)
    worksheet.insert_image('B2', pic_path)
    writer.save()
    

# 以下用于模拟报告
def gen_pr_od(df, score='score', label='7d'):
    """生成通过率对应的逾期率"""
    od_list = []
    for i in range(100):
        th = np.percentile(df[score].astype(float), i)
        od = df.query(f'{score} > {th}')[label].agg([len, np.sum, np.mean]).tolist()
        od = [100 - i, th] + od
        od_list += [od]
        
    return pd.DataFrame(od_list, columns=['passing_ratio', 'threshold', 'cnt', 'sum', 'overdue_ratio'])
    tmp = pd.DataFrame({'passing_ratio' : pr_list, 'overdue_ratio' : od_list, 'threshold' : th_list})
    return tmp[['passing_ratio', 'threshold', 'overdue_ratio']]  


def _get_common_col(df1, df2):
    """获取col的交集"""
    return sorted([fea for fea in df1.columns if fea in df2.columns])


def _get_common_idx(df1, df2, idx=None):
    """idx的交集，可以为index，也可以为某一给定的col"""
    idx1 = df1.index.tolist() if idx is None else df1[idx].tolist()
    idx2 = df2.index.tolist() if idx is None else df2[idx].tolist()
    
    if (len(idx1) != len(set(idx1))) or (len(idx2) != len(set(idx2))):
        msg = 'index of dataframe is not unique!' if idx is None else f'{idx} of dataframe is not unique!'
        raise Exception(msg)
    return sorted(set(idx1).intersection(set(idx2)))


def get_common_data(df1, df2, columns=None, index=None, precison=3):
    """获取df1和df2的交集数据"""
    com_col = _get_common_col(df1, df2) if columns is None else columns
    com_idx = _get_common_idx(df1, df2, index)
    
    if index is not None:
        df_1 = df1.set_index(index)
        df_2 = df2.set_index(index)
        com_col = [col for col in com_col if col != index]
    else:
        df_1 = df1.copy()
        df_2 = df2.copy()

    df11 = df_1.loc[com_idx][com_col].apply(pd.to_numeric, errors='ignore').round(precison)
    df22 = df_2.loc[com_idx][com_col].apply(pd.to_numeric, errors='ignore').round(precison)
    return df11, df22


def _get_consistency_ratio_under_tolerance(df1, df2, tolerance=0.05, smooth=0.001, how='left'):
    """对比df1和df2两个df，误差在容忍度之内可视为一致，以左边/右边的数据为准
    :param df1:
    :param df2:
    :param tolerance:
    :param how: 'left', 'right'
    """
    diff_dict = {}
    for col in df1.columns:
        if (df1[col].dtypes == np.object) or (df2[col].dtypes == np.object):
            diff_dict[col] = (df1[col] == df2[col]).agg([np.mean, np.sum]).tolist()
        else:
            base = df1[col] if how == 'left' else df2[col]
            diff_dict[col] = (abs((df1[col] - df2[col]) / (base + smooth)) < tolerance).agg([np.mean, np.sum]).tolist()
    return pd.DataFrame(diff_dict, index=[f'容忍度为{tolerance}的一致率', f'容忍度为{tolerance}的一致个数']).T

    
def get_consistency_ratio(df1, df2, fillna_val=None, tolerance=0.05, smooth=0.001, how='left'):
    """对比两个dataframe的一致率，并对比用某些值填充后的一致率
    :param df1: 
    :param df2: 
    :param fill_na_val:
    :return:
    """
    tmp1 = pd.DataFrame(df1 == df2).agg([len, np.mean, np.sum]).T
    tmp1.columns = ['总个数', '实际一致率', '实际一致个数']
    
    if fillna_val is not None:
        tmp2 = pd.DataFrame(df1.fillna(fillna_val) == df2.fillna(fillna_val)).agg([np.mean, np.sum]).T
        tmp2.columns = [f'空值填{fillna_val}的一致率', f'空值填{fillna_val}的一致个数']

    tmp3 = _get_consistency_ratio_under_tolerance(df1.fillna(fillna_val), df2.fillna(fillna_val), 
                                                  tolerance=tolerance, smooth=smooth, how=how)
    
    tmp3.columns = [f'空值填{fillna_val},容忍度为{tolerance}的一致率', f'空值填{fillna_val},容忍度为{tolerance}的一致个数']
    
    tmp = tmp1.join(tmp3) if fillna_val is None else tmp1.join(tmp2).join(tmp3)
    for col in tmp.columns:
        if col.endswith('个数'):
            tmp[col] = tmp[col].astype(int)
    return tmp
    
def get_difference_values(df1, df2, name1='online_data', name2='offline_data', idx='apply_risk_id'):
    """对比df1和df2，并列出不同的值
    :param df1: 
    :param df2: 
    :param name1:
    :param name1:
    :param idx:
    :return:
    """
    from collections import defaultdict
    df_diff = pd.DataFrame(df1 == df2)
    df_diff2 = df_diff.copy()
    idx_dic = defaultdict(list)
    for col in df_diff.columns:
        idx_dic[col] = df_diff.query(f'{col} == 0').index.tolist()
        
    for col, idx_list in idx_dic.items():
        df_tmp = pd.DataFrame()
        df_tmp[idx] = idx_list
        df_tmp[name1] = df1.loc[idx_list][col].tolist()
        df_tmp[name2] = df2.loc[idx_list][col].tolist()
        df_tmp['diff_data'] = name1 + ':' + df_tmp[name1].astype(str) + ', ' + name2 + ':' + df_tmp[name2].astype(str)
        df_tmp.set_index(idx, inplace=True)
        df_diff[col].loc[idx_list] = df_tmp['diff_data'].loc[idx_list]
    return df_diff, df_diff2


def get_desc_report(df1, df2, label=None):
    """调用desc_report中的func
    :param df1:
    :param df2:
    :param label:
    """
    return mlearn.service.reporter.data_reporter._gen_desc_report(df1, df2, label, '')


def sort_app_names(df, col_name='equipment_app_names_v2', sep='|', drop_duplicates=False):
    """对app的那列分割后重新排序
    :param df:
    :param col_name:
    :param sep:
    :param drop_duplicates:
    """
    def _sort_app_names(x, sep='|', drop_duplicates=False):
        xx = str(x).split(sep)
        xx2 = sep.join(sorted(set(xx))) if drop_duplicates else sep.join(sorted(xx))
        return xx2
    df2 = df.copy()
    df2[col_name] = df2[col_name].map(lambda x : _sort_app_names(x, sep=sep, drop_duplicates=drop_duplicates))
    return df2


def gen_compare_report(df1, df2, columns=None, index=None, precison=3, fillna_val=9999, 
                       app_col=None, app_sep='|', app_drop_dup=False):
    """获取df1和df2的对比报告，需保证idx的唯一性
    :param df1: 
    :param df2: 
    :param columns:
    :param index:
    :param precison:
    :param app_col:
    :param app_sep:
    :param app_drop_dup:
    :return:
    """
    df11, df22 = get_common_data(df1, df2, columns=columns, index=index, precison=precison)
    if app_col is not None:
        df11 = sort_app_names(df11, col_name=app_col, sep=app_sep, drop_duplicates=app_drop_dup)
        df22 = sort_app_names(df22, col_name=app_col, sep=app_sep, drop_duplicates=app_drop_dup)
        
    df_consis_report = get_consistency_ratio(df11, df22, fillna_val=fillna_val)
    df_diff_report1, df_diff_report2 = get_difference_values(df11.fillna(fillna_val), df22.fillna(fillna_val))
    # df_desc_report = get_desc_report(df11, df22)
    df_desc_report = None
    return df_consis_report, df_diff_report1, df_diff_report2, df_desc_report


def dfs_to_excel(df_dic, file_name=None, index=True):
    """将dic中的df写入同一个excel
    :param df_dic:
    :param file_name:
    :param index:
    """
    import time, xlsxwriter
    if file_name is None:
        file_name = time.strftime("%Y%m%d", time.localtime()) + '_Report.xlsx'

    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    for name, df in df_dic.items():
        df.to_excel(writer, sheet_name=name, index=index)
    writer.save()


def _get_auc_ks_mean(df, cols=['14d', 'score']):
    ks, ac, pr, rc, fpr, tpr, thr_point = cal_metrics(df[cols[0]], df[cols[1]])
    if ac < 0.5:
        ks, ac, pr, rc, fpr, tpr, thr_point = cal_metrics(df[cols[0]], -df[cols[1]])
    mean = df[cols[0]].mean()
    return ks, ac, mean


def cal_level_distribution(df, target='score', agg_col='apply_day', span=7):
    """
    df:
    target:
    agg_col:
    span:
    
    e.g. cal_level_distribution(df_tmp, span=14)
    """
    unique_list = sorted(df[agg_col].unique().tolist())
    ascending=False
    
    mm = ModelMonitor()
    cp = [350] + mm.get_cut_points_by_freq(df[target])[1:-1] + [950]
    
    result = []
    for idx_s in list(range(0, len(unique_list) - span)):
        idx_e = min(idx_s + span, len(unique_list) - 1)
        time_span = unique_list[idx_s : idx_e]
        df_tmp = df[df[agg_col].isin(time_span)]
        df_tmp['level'] = pd.cut(df_tmp[target], bins=cp, labels=list(range(10, 0, -1)))
        dis = (df_tmp['level'].value_counts().sort_index(ascending=ascending) / df_tmp.shape[0]).tolist()
        idx = df_tmp['level'].value_counts().sort_index(ascending=ascending).index.tolist()
        result.append([unique_list[idx_s] + '~' + unique_list[idx_e]] + dis)
    return_cols = ['apply_span'] + [f'level_{i}' for i in idx]
    return pd.DataFrame(result, columns=return_cols)


def cal_rolling(df, metric, metric_args, return_cols=['apply_span', 'ks', 'auc', 'overdue'], agg_col='apply_day', span=7):
    """
    df:
    metric:
    metric_args:
    return_cols:
    agg_col:
    span:
    
    e.g. cal_rolling(df_tmp, _get_auc_ks_mean, ['14d', 'score'], span=14)
    """
    unique_list = sorted(df[agg_col].unique().tolist())
    metric_vals = []
    for idx_s in list(range(0, len(unique_list) - span)):
        idx_e = min(idx_s + span, len(unique_list) - 1)
        time_span = unique_list[idx_s : idx_e]
        df_tmp = df[df[agg_col].isin(time_span)]
        
        ks, ac, mean = metric(df_tmp, metric_args)
        metric_vals.append([unique_list[idx_s] + '~' + unique_list[idx_e], ks, ac, mean])
    return pd.DataFrame(metric_vals, columns=return_cols)

