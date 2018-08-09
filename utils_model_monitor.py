import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')

from .utils_bins import Bins, WOE

__ALPHA__ = 0.9


class ModelMonitor(Bins):
    def __init__(self):
        pass
    
    
    def plot_descrete_single_value(self, df_all, title, ax=None):
        from matplotlib.ticker import FuncFormatter 
        def to_percent(temp, position):  
            return '%1.0f'%(100 * temp) + '%' 

        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
        
        df_tmp = df_all.iloc[:, 1:]
        feature_value = df_all.iloc[:, 0].tolist()
        df_tmp.plot.bar(title=title, ax=ax, alpha=__ALPHA__)
        ax.set_xticklabels(feature_value, rotation=45, horizontalalignment='right')
        ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax.set_title(title)


    def plot_discrete_values(self, df_dic, cols, path='cat_val.png'):
        '''
        Args:
            df_dic: 
            cols: features
            path: path for save

        Example:
            plot_discrete_values({'OOT' : df_res, 'Online' : df_oot, 'Online2' : df_oot}, cols, path='cat_val.png')
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
                            left, right, on='feature', how='outer'), list_tmp)
            df_tmp.sort_values(by='feature', inplace=True)
            sub_ax = axs[i] if num_cols > 1 else axs
            self.plot_descrete_single_value(df_tmp, col, ax=sub_ax)
            plt.tight_layout()
        plt.savefig(path)
        plt.show()


    def plot_continuous_single_value(self, x, title, bins=100, ax=None):
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
        ax.hist(x, bins=bins, alpha=__ALPHA__)
        ax.set_title(title)


    def plot_continuous_values(self, df_dic, cols, is_bins=False, path='cont_val.png'):
        '''
        Args:
            df_dic: 
            cols: features
            path: path for save

        Example:
            plot_continuous_values({'OOT': df_oot, 'Online': df_res, 'Online2': df_res}, cols, is_bins=True, path='cont_val_bins.png')
            plot_continuous_values({'OOT': df_oot, 'Online': df_res, 'Online2': df_res}, cols, path='cont_val.png')
        '''
        from functools import reduce
        num_cols, num_dics = len(cols), len(df_dic.keys())

        # bin for numeric values
        if is_bins:
            num_cat = 1 if is_bins else 2
            base_key = list(df_dic.keys())[0]

            fig, axs = plt.subplots(num_cols, num_cat, figsize=(8 * num_cat, 6 * num_cols))
            for i, col in enumerate(cols):
                x = df_dic[base_key][col].dropna()
                cp = self.get_cut_points_by_freq(x)
                list_tmp = []
                for j, key in enumerate(df_dic.keys()):
                    df_tmp = df_dic[key]
                    x = df_tmp[df_tmp[col].notnull()][col]
                    x_bin = self.bins(x, cut_points=cp).bucket.value_counts(normalize=True)
                    x_bin = pd.DataFrame(x_bin.reset_index(drop=False))
                    x_bin.columns = ['feature', f'{col}_{key}']
                    list_tmp.append(np.round(x_bin, 5))
                    
                df_tmp = reduce(lambda left, right: pd.merge(left, right, on='feature', how='outer'), list_tmp)
                df_tmp.sort_values(by='feature', inplace=True)
                sub_ax = axs[i] if num_cols > 1 else axs
                self.plot_descrete_single_value(df_tmp, col, ax=sub_ax)
                plt.tight_layout()
        else:
            num_cat = len(df_dic)
            fig, axs = plt.subplots(num_cols, num_cat, figsize=(8 * num_cat, 6 * num_cols))
            for i, col in enumerate(cols):
                for j, key in enumerate(df_dic.keys()):
                    df_tmp = df_dic[key]
                    x = df_tmp[df_tmp[col].notnull()][col]
                    if num_dics == 1 and num_cols == 1:
                        sub_ax = axs
                    elif num_dics > 1:
                        sub_ax = axs[i][j]
                    else:
                        sub_ax = axs[i]
                    self.plot_continuous_single_value(x, f'{col}_{key}', bins=100, ax=sub_ax)
            plt.tight_layout()

        plt.savefig(path)
        plt.show()



    def plot_single_value_with_label(self, df_bin, title, ax=None):
        from matplotlib.ticker import FuncFormatter 
        def to_percent(temp, position):  
            return '%1.0f'%(100 * temp) + '%' 

        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
        ax2 = ax.twinx()

        df_tmp = df_bin.groupby('bucket', as_index=True)
        df_bucket = pd.DataFrame()
        df_bucket['count_y'] = df_tmp['y'].count() / len(df_bin)
        df_bucket['mean_y'] = df_tmp['y'].mean()
        df_bucket = df_bucket.reset_index(drop=False).sort_values(by='bucket')
        df_bucket.columns = ['feature', 'proportion', 'mean_label']
        
        ax.bar(range(df_bucket.shape[0]), df_bucket['proportion'], 
               tick_label=df_bucket['feature'], label='volume', alpha=__ALPHA__)
        ax2.plot(range(len(df_bucket)), df_bucket['mean_label'], label='mean_label',
                marker='o', markerfacecolor='r', c='black')
        ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax2.yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax.set_xticklabels(df_bucket['feature'], rotation=45, horizontalalignment='right')
        ax2.legend(loc='best')
        ax.set_title(title)


    # model_monitor.plot_values_with_label??
    def plot_values_with_label(self, df_dic, cols, isolate=False, path='vals_with_label.png', n_dis=10):
        from pandas.api.types import is_numeric_dtype
        num_cols, num_dics = len(cols), len(df_dic.keys())
        fig, axs = plt.subplots(num_cols, num_dics, figsize=(8 * num_dics, 6 * num_cols))
        base_key = list(df_dic.keys())[0]
        for i, col in enumerate(cols):
            print('>>> col:', col)
            x = df_dic[base_key][0][col].dropna()
            if is_numeric_dtype(x) and len(set(x)) >= n_dis:
                # numeric values
                if not isolate:
                    cp = self.get_cut_points_by_freq(x)
                list_tmp = []
                for j, key in enumerate(df_dic.keys()):
                    x, y = df_dic[key]
                    y = y[x[col].notnull()]
                    x = x[x[col].notnull()][col]
                    if isolate:
                        cp = self.get_cut_points_by_freq(x)
                    df_bin = self.bins(x, y, cut_points=cp).reset_index(drop=False)   
                    if num_dics == 1 and num_cols == 1:
                        sub_ax = axs
                    elif num_dics > 1 and num_cols == 1:
                        sub_ax = axs[j]
                    elif num_dics == 1 and num_cols > 1:
                        sub_ax = axs[i]
                    else:
                        sub_ax = axs[i][j]
                    self.plot_single_value_with_label(df_bin, col + '_' + key, ax=sub_ax)
                plt.tight_layout()

            else:
                # discrete values
                list_tmp = []
                for j, key in enumerate(df_dic.keys()):
                    x, y = df_dic[key]
                    y = y[x[col].notnull()]
                    x = x[x[col].notnull()][col]
                    df_bin = pd.DataFrame({'bucket' : x, 'y' : y})
                    if num_dics == 1 and num_cols == 1:
                        sub_ax = axs
                    elif num_dics > 1 and num_cols == 1:
                        sub_ax = axs[j]
                    elif num_dics == 1 and num_cols > 1:
                        sub_ax = axs[i]
                    else:
                        sub_ax = axs[i][j]
                    self.plot_single_value_with_label(df_bin, col + '_' + key, ax=sub_ax)
                    plt.tight_layout()
            plt.savefig(path)
        

    def data_describe(self, df_dic, cols, path=''):
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


    def plot_single_trend(self, df_bins, mean, title, ax=None):
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


    def plot_trends(self, df_meta, cols, interval='day', time_col='apply_time_commit_at', path='data_trends.png'):
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
            self.plot_single_trend(tmp, mean, col, ax=axs[i])
            plt.tight_layout()
        plt.savefig(path)



    def get_psi(self, x_actual, x_expected, num_of_bins=10, psi_only=True):
        from math import log, e
        cp = self.get_cut_points_by_freq(x_actual)
        x_bin1 = pd.DataFrame(self.bins(x_actual, cut_points=cp).bucket.value_counts(normalize=True))
        x_bin1.columns = ['bucket_expected']
        x_bin2 = pd.DataFrame(self.bins(x_expected, cut_points=cp).bucket.value_counts(normalize=True))
        x_bin2.columns = ['bucket_actual']

        bin_tmp = x_bin1.join(x_bin2)
        bin_tmp['prop_diff'] = bin_tmp['bucket_actual'] - bin_tmp['bucket_expected']
        bin_tmp['prop_ln'] = (bin_tmp['bucket_actual'] / bin_tmp['bucket_expected']).map(lambda x: log(x, e))
        bin_tmp['psi'] = bin_tmp['prop_diff'] * bin_tmp['prop_ln']
        if psi_only:
            return bin_tmp['psi'].sum()
        else: 
            return bin_tmp




if __name__ == '__main__':
    val_count = df_oot[fea_in].nunique()
    fea_continuous = val_count[val_count > 10].index.tolist()
    fea_discrete = val_count[val_count <= 10].index.tolist()

    # 统计数据
    df_describe = data_describe({'OOT' : df_oot, 'Online' : df_res}, fea_in, 'test_report.xlsx')

    # 画离散值
    cols = ['r_woe_province', 'r_woe_xy_score']
    plot_discrete_values({'OOT' : df_res, 'Online' : df_oot, 'Online2' : df_oot}, cols)

    # 画连续值
    plot_continuous_values({'OOT' : df_oot, 'Online' : df_res, 'Online2' : df_res}, fea_continuous)

    # 画连续值 - 分箱
    cols = ['kx_score', 'jd_score', 'r_woe_xy_score']
    plot_continuous_values({'OOT': df_oot, 'Online': df_res, 'Online2': df_res}, cols, is_bins=True, path='cont_val_bins.png')
