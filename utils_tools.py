import pandas as pd
import numpy as np
import re, math, hashlib, json


def get_md5(string):
    '''
    加密函数
    '''
    return hashlib.md5(str(string).encode('utf-8')).hexdigest()


def filter_numbers(n):
    '''
    过滤清洗手机号码
    '''
    pattern = re.compile('(.*)?(1(34|35|36|37|38|39|47|50|51|52|57|58|59|72|78|82|83|84|87|88|98|30|31|32|45|55|56|66|71|75|76|85|86|33|49|53|73|77|80|81|89|99|70)\d{8})(\.0)?$')
    m = pattern.match(n)
    if m:
        return m.groups()[1]
    else:
        return n[:-2] if n.endswith('.0') else n


def format_json(x, null_content):
    if pd.isnull(x):
        # null_content = '{"query_times": "{\\"7d\\": 0, \\"15d\\": 0, \\"30d\\": 0, \\"3m\\": 0, \\"6m\\": 0, \\"12m\\": 0, \\">12m\\": 0}", "query_org_cnt": "{\\"7d\\": 0, \\"15d\\": 0, \\"30d\\": 0, \\"3m\\": 0, \\"6m\\": 0, \\"12m\\": 0, \\">12m\\": 0}", "query_org_types": "[]"}'
        return null_content
    else:
        x = json.loads(x)
        for k, v in x.items():
            x[k] = json.dumps(v)
        return json.dumps(x)


def json_to_df(x):
    import json
    x = x.apply(lambda s:pd.Series(json.loads(s)))
    return x.replace('', np.nan).convert_objects(convert_numeric=True)


def get_types(df, fea_out=None):
    from pandas.api.types import is_numeric_dtype
    fea_num = df.dtypes[df.dtypes.apply(is_numeric_dtype)].index.tolist()
    fea_str = df.dtypes[~df.dtypes.apply(is_numeric_dtype)].index.tolist()
    if fea_out:
        fea_num = [fea for fea in fea_num if fea not in fea_out]
        fea_str = [fea for fea in fea_str if fea not in fea_out]
    return fea_num, fea_str


def prob_to_score(p, A=423.82, B=72.14):
    """
    args:
        p: 模型输出的概率，对1的概率
        A: 基础分补偿，不用修改
        B: 刻度，不用修改
    return:
        score: 分数，头尾掐掉
    """
    odds = p / (1 - p)
    score = A - B * math.log(odds)
    score = max(350, score)
    score = min(950, score)
    return score


def sigmoid(logit):
    """
    args:
        logit: logistics model 输出的值
    return:
        激活函数返回的值
    """
    return 1.0 / (1 + math.exp(-logit))


def score_to_risklevel(score, cut_points=None):
    """
    args:
        score: 由概率转换出的分数值
        cut_points: 十等分切割点
    return:
        risk_level: 返回一个风险等级
    """
    if not cut_points:
        cut_points = [350, 410, 470, 530, 590, 650, 710, 770, 830, 890, 950]
    return int(pd.cut([score], bins=cut_points, labels=list(range(10, 0, -1)), include_lowest=True)[0])


import re
def get_phone_reg_time(x):
    r = []
    if not pd.isnull(x):
        r = re.findall(r'使用(\d+)个月', x)
    return r[0] if len(r) > 0 else np.nan


import json
def extract_nested_json(x):
    """
    args:
        x: json-formated data
    return:
        anti-nested formated data 将多层嵌套的json提取出只有一层的json，返回数据也是json类型
    example:
        df.data.map(extract_nested_json).apply(lambda s:pd.Series(json.loads(s)))
    """
    global_dic = {}
    x = x.replace('\\\\', '\\')
    def json_to_dict(key, value, prefix=''):
        if isinstance(value, dict):
            for k, v in value.items():
                if key and prefix:
                    json_to_dict(k, v, prefix + '_' + key)
                elif key and not prefix:
                    json_to_dict(k, v, key)
                elif not key and prefix:
                    json_to_dict(k, v, prefix)
                else:
                    json_to_dict(k, v, '')
        else:
            if prefix:
                key = prefix + '_' + key
            global_dic[key] = value
    try:
        tmp = json.loads(x)
        json_to_dict('', tmp)
    except:
        global_dic['_ERROR_'] = 1
    return json.dumps(global_dic)



def stat_data(df_meta, col, target='14d'):
    df = df_meta.copy()
    df['c'] = 1
    tmp0 = pd.DataFrame(df.groupby(col)[target].mean()).reset_index(drop=False)
    tmp0.columns = [col, '违约率']
    tmp1 = pd.DataFrame(df.groupby(col)[target].sum()).reset_index(drop=False)
    tmp1.columns = [col, '违约样本量']
    tmp2 = pd.DataFrame(df.groupby(col)['c'].sum()).reset_index(drop=False)
    tmp2.columns = [col, '样本量']

    tmp = pd.merge(tmp2, tmp1, on=col)
    tmp['非违约样本量'] = tmp['样本量'] - tmp['违约样本量']
    tmp = pd.merge(tmp, tmp0, on=col)
    return tmp


def sample(df, overduerate=0.17, target='14d', random_state=2017):
    """
    Args:
        df:
        overduerate:
        target:
        random_state:

    Return:
        df
    """
    overduerate1 = df[target].mean()

    if overduerate > overduerate1:
        # 逾期率升高, 保留坏用户, 去掉一些好用户
        df_good = df[df[target] == 0]
        df_bad = df[df[target] == 1]
        n_bad = df_bad.shape[0]
        n_good = int(((1 - overduerate) / overduerate) * n_bad)
        df_good = df_good.sample(n=n_good, random_state=random_state)
        df2 = pd.concat([df_good, df_bad])

    elif overduerate < overduerate1:
        # 逾期率降低, 保留好用户, 去掉一些坏用户
        df_good = df[df[target] == 0]
        df_bad = df[df[target] == 1]
        n_good = df_good.shape[0]
        n_bad = int((overduerate / (1 - overduerate)) * n_good)
        df_bad = df_bad.sample(n=n_bad, random_state=random_state)
        df2 = pd.concat([df_good, df_bad], axis=0)
    else:
        return df
    return df2


def _get_random_time(start=1510987104669, end=1540987104669, seed=None):
    import random, time
    if not seed:
        t = random.uniform(start, end)
    else:
        random.seed(seed)
        t = random.uniform(start, end)
    dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t / 1000))
    return dt

def get_random_time(start=1510987104669, end=1540987104669, prefix=''):
    dt = _get_random_time(start, end)
    if prefix:
        while not dt.startswith(prefix):
            dt = _get_random_time(start, end)
    return dt
