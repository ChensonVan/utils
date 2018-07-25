from .utils_tools import *

def features_analyse(pyce, df, num_fea, fea_str, target='14d', prefix='Feature_analyse_', 
                     save_result=True, monotonic_bin=True, prof_tree_cut=True, prof_min_p=0.05,
                     prof_cut_group=10, max_missing_rate=0.95, event=1, prof_threshold_cor=1, 
                     exec_recoding=False):
    recoding_prefix = 'r_'
    out_feature_recoding = prefix + 'features_recoding.txt'

    # output feature profiling to file
    out_features_profile = prefix + 'features_profile.csv'

    # output feature statistics to file
    out_features_statistics = prefix + 'features_statistics.csv'

    df_profile, df_statistics, statement_recoding = pyce.features_prof_recode(
                                                            Xcont=df[num_fea],  # set as pd.DataFrame() if none
                                                            Xnomi=df[fea_str],  # set as pd.DataFrame() if none
                                                            Y=df[target],       # Y will be cut by median if non-binary target
                                                            event=event, 
                                                            max_missing_rate=max_missing_rate,
                                                            recoding_std=False,
                                                            recoding_woe=True,
                                                            recoding_prefix=recoding_prefix,
                                                            prof_cut_group=prof_cut_group,
                                                            monotonic_bin=monotonic_bin,
                                                            prof_tree_cut=prof_tree_cut,
                                                            prof_min_p=prof_min_p,
                                                            prof_threshold_cor=prof_threshold_cor,
                                                            class_balance=True)

    """
    Information Value:
    < 0.02: useless for prediction
    0.02~0.1: Weak predictor
    0.1~ 0.3: Medium predictor
    0.3~0.5: Strong predictor 
    >0.5: Suspicious or too good to be true
    """

    if save_result:
        pyce.write_recoding_txt(statement_recoding, file=out_feature_recoding, encoding='utf-8')
        df_profile.to_csv(out_features_profile, encoding='gbk', index=False)
        df_statistics.to_csv(out_features_statistics, encoding='gbk', index=False)
    print('INFO : Analyse Finished.')
    
    if exec_recoding:
        data_recoded = pyce.exec_recoding(df, recoding_txt=out_feature_recoding, encoding='utf-8')
        print('INFO : Exec_Recoding Finished.')
        return data_recoded
    else:
        return df
    
    


def LR_predict(df, get_feature, woe_encoder, weight_list, fea_list):
    df2 = get_feature(df)
    df2 = woe_encoder(df2)
    df2['y_pred'] = (weight_list * df2[fea_list]).apply(sum, axis=1)
    df2['y_pred'] = round(df2['y_pred'].map(sigmoid), 6)
    df2['score'] = round(df2['y_pred'].map(prob_to_score), 4)
    return df2


def LR_fit(pyce, X, y, prefix='', fea_selection=True, save_result=False, num_var_final=20, stepwise=True):
    statmodel_significant_level = 0.05
    X['intercept'] = 1

    if fea_selection:
        statmodel_significant_vars = pyce.feature_selection_logistic(y, X, feature_n=num_var_final, 
                                                                     alpha=statmodel_significant_level, 
                                                                     stepwise=stepwise)
        indep_var_final = statmodel_significant_vars
    else:
        indep_var_final = X.columns.tolist()
    
    import statsmodels.api as sm
    model_desc = sm.Logit(y, X[indep_var_final])  
    model_final = model_desc.fit(disp=False)

    model_summary = model_final.summary2(alpha=0.05)
    model_describe = pd.DataFrame()
    model_describe['estimate'] = model_final.params
    model_describe['stderror'] = model_final.bse
    model_describe['tvalues'] = model_final.tvalues
    model_describe['pvalues'] = model_final.pvalues
    features_vif = pyce.VIF(Y=y, X=X[indep_var_final])
    model_describe = model_describe.join(features_vif)

    if save_result:
        model_final.save(prefix + "final_model.pkl")
        model_describe.to_csv(prefix + "Final_Model_Summary.csv", encoding='utf-8')

    return model_describe, model_final