import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')



def get_cut_points(data, num_of_bins=10):
    try:
        cut_points = list(pd.qcut(data, q=np.linspace(0, 1, num_of_bins+1), precision=10, retbins=True, duplicates='drop')[1])[1 : -1]    
    except:
        print('Error')
    return cut_points


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


def plot_descrete_value_single(df_all, title, ax=None):
    from matplotlib.ticker import FuncFormatter 
    def to_percent(temp, position):  
        return '%1.0f'%(100 * temp) + '%' 
    
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    df_tmp = df_all.iloc[:, 1:]
    feature_value = df_all.iloc[:, 0].tolist()
    df_tmp.plot.bar(title=title, ax=ax)
    ax.set_xticklabels(feature_value, rotation=45, horizontalalignment='right')
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax.set_title(title)


def plot_discrete_value(df_dic, cols, path='cat_val.png'):
    '''
    Args:
        df_dic: 
        cols: features
        path: path for save

    Example:
        plot_discrete_value({'OOT' : df_res, 'Online' : df_oot, 'Online2' : df_oot}, cols, path='cat_val.png')
    '''
    from functools import reduce
    num_cols = len(cols)
    fig, axs = plt.subplots(num_cols, 1, figsize=(8, 6 * num_cols))

    for i, col in enumerate(cols):
        list_tmp = []
        for j, key in enumerate(df_dic.keys()):
            df_tmp = df_dic[key]
            x = df_tmp[df_tmp[col].notnull()][col]
            x = pd.DataFrame(x.value_counts(normalize=True)).reset_index(drop=False)
            x.columns = ['feature', f'{col}_{key}']
            try:
                list_tmp.append(np.round(x, 5))
            except:
                list_tmp.append(x)

        df_tmp = reduce(lambda left, right: pd.merge(
                        left, right, on='feature'), list_tmp)
        df_tmp.sort_values(by='feature', inplace=True)
        sub_ax = axs[i] if num_cols > 1 else axs
        plot_descrete_value_single(df_tmp, col, ax=sub_ax)
        plt.tight_layout()
    plt.savefig(path)
    plt.show()


    
def plot_continuous_value_single(x, title, bins=100, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    ax.hist(x, bins=bins)
    ax.set_title(title)

    

def plot_continuous_value(df_dic, cols, is_bins=False, path='cont_val.png'):
    '''
    Args:
        df_dic: 
        cols: features
        path: path for save

    Example:
        plot_continuous_value({'OOT': df_oot, 'Online': df_res, 'Online2': df_res}, cols, is_bins=True, path='cont_val_bins.png')
        plot_continuous_value({'OOT': df_oot, 'Online': df_res, 'Online2': df_res}, cols, path='cont_val.png')
    '''
    from functools import reduce
    num_cols = len(cols)

    # 对连续值分箱
    if is_bins:
        num_cat = 1 if is_bins else 2
        base_key = list(df_dic.keys())[0]

        fig, axs = plt.subplots(num_cols, num_cat, figsize=(8 * num_cat, 6 * num_cols))
        for i, col in enumerate(cols):
            x = df_dic[base_key][col].dropna()
            cp = get_cut_points(x)
            list_tmp = []
            for j, key in enumerate(df_dic.keys()):
                df_tmp = df_dic[key]
                x = df_tmp[df_tmp[col].notnull()][col]
                x_bin = bins_points(x, cp)
                x_bin = pd.DataFrame(x_bin.value_counts(normalize=True)).reset_index(drop=False)
                x_bin.columns = ['feature', f'{col}_{key}']
                list_tmp.append(np.round(x_bin, 5))
            df_tmp = reduce(lambda left, right: pd.merge(left, right, on='feature'), list_tmp)
            df_tmp.sort_values(by='feature', inplace=True)
            sub_ax = axs[i] if num_cols > 1 else axs
            plot_descrete_value_single(df_tmp, col, ax=sub_ax)
            plt.tight_layout()
    else:
        num_cat = len(df_dic)
        fig, axs = plt.subplots(num_cols, num_cat, figsize=(8 * num_cat, 6 * num_cols))
        for i, col in enumerate(cols):
            for j, key in enumerate(df_dic.keys()):
                df_tmp = df_dic[key]
                x = df_tmp[df_tmp[col].notnull()][col]
                sub_ax = axs[i][j] if num_cols > 1 else axs[j]
                plot_continuous_value_single(x, f'{col}_{key}', bins=100, ax=sub_ax)
        plt.tight_layout()

    plt.savefig(path)
    plt.show()

    

def data_describe(df_dic, cols, path=''):
    '''
    Args:
        df_dic:
        cols:
    
    Example:
        data_describe({'OOT' : df_oot, 'Online' : df_res}, fea_in, 'Data_describe.xlsx')
    '''
    from functools import reduce
    fea = ['count', 'cov_rate', 'mean', 'std', 'max', 'min', '25%', '50%', '75%']
    fea_out = ['index'] + [f'{f1}_{f2}' for f1 in fea for f2 in df_dic.keys()]
            
    list_tmp = []
    for i, key in enumerate(df_dic.keys()):
        df_tmp = df_dic[key].describe().T
        df_tmp.columns = df_tmp.columns + '_' + key
        df_tmp[f'cov_rate_{key}'] = df_dic[key][cols].notnull().mean()
        df_tmp.reset_index(inplace=True)
        list_tmp.append(df_tmp)
    df_tmp = reduce(lambda left, right: pd.merge(left, right, on='index'), list_tmp)
    if path:
        df_tmp[fea_out].to_excel(path, index=False, encoding='gbk')
    return df_tmp[fea_out]



def plot_trend_single(df_bins, mean, title, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    x = df_bins.iloc[:, 0]
    y = df_bins.iloc[:, 1]
    
    ax.plot([i for i in range(len(x))], y, label='Online',
            marker='o', markerfacecolor='r')
    ax.axhline(mean, color='k', label='Baseline', linestyle='-.')
    ax.set_xticks([i for i in range(len(x))])
    ax.set_xticklabels(x, rotation=40)
    ax.set_title(title)
    ax.legend()

def plot_trends(df_meta, cols, interval='day', time_col='apply_time_commit_at', path='data_trends.png'):
    df = df_meta.copy()
    if interval == 'day':
        df['commit_time'] = pd.to_datetime(df[time_col]).astype(str).str[:10]
    elif interval == 'hour':
        df['commit_time'] = pd.to_datetime(df[time_col]).dt.hour
    elif interval == 'day_hour':
        df['commit_time'] = pd.to_datetime(df[time_col]).astype(str).str[:13]
    else:
        print('_ERROR_')
        return

    num_cols = len(cols)
    width = np.round(df['commit_time'].nunique() / 12)
    width = min(max(width, 1), 2)
    fig, axs = plt.subplots(num_cols, 1, figsize=(10 * width, 6 * num_cols))

    for i, col in enumerate(cols):
        if df[col].nunique() == 0:
            continue
        tmp = pd.DataFrame(df.groupby('commit_time')[col].mean())
        # 取第一条数据为baseline
        # mean = tmp.iloc[0][0]
        # 取整体平均值为baseline
        mean = tmp.mean()[0]
        tmp.reset_index(drop=False, inplace=True)
        plot_trend_single(tmp, mean, col, ax=axs[i])
        plt.tight_layout()
    plt.savefig(path)



if __name__ == '__main__':
    val_count = df_oot[fea_in].nunique()
    fea_continuous = val_count[val_count > 10].index.tolist()
    fea_discrete = val_count[val_count <= 10].index.tolist()

    # 统计数据
    df_describe = data_describe({'OOT' : df_oot, 'Online' : df_res}, fea_in, 'test_report.xlsx')

    # 画离散值
    cols = ['r_woe_province', 'r_woe_xy_score']
    plot_discrete_value({'OOT' : df_res, 'Online' : df_oot, 'Online2' : df_oot}, cols)

    # 画连续值
    plot_continuous_value({'OOT' : df_oot, 'Online' : df_res, 'Online2' : df_res}, fea_continuous)

    # 画连续值 - 分箱
    cols = ['kx_score', 'jd_score', 'r_woe_xy_score']
    plot_continuous_value({'OOT': df_oot, 'Online': df_res, 'Online2': df_res}, cols, is_bins=True, path='cont_val_bins.png')
