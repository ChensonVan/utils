import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

class BasicModel(object):
    """ Parent class of basic models """
    @abstractmethod
    def train(self, x_train, y_train, x_val, y_val, params=None, is_eval=False, is_verbose=False):
        """ return the predicted result of test data """
        print('INFO  : Training Model ... ...')
        self.init_model(params)

        # print(self.model)

        if is_eval:
            self.model.fit(x_train, y_train.ravel()
                           ,eval_set = [(x_train, y_train), (x_val, y_val)]
                           ,eval_metric=self.metric
                           ,early_stopping_rounds=self.early_stopping_rounds
                           ,verbose=is_verbose
                           )
        else:
            self.model.fit(x_train, y_train.ravel())

        y_pred = self.predict(x_val)
        auc_score = metrics.roc_auc_score(y_val,  y_pred)
        return auc_score


    def predict(self, x_test):
        """ return the predicted result of test data """
        if self.model == None:
            raise Exception('Please fit the data by using model before predict') 
        try:
            y_pred = self.model.predict_proba(x_test)[:, 1]
        except:
            y_pred = self.model.predict(x_test)
        return y_pred


    @abstractmethod
    def init_model(self, params):
        pass


    def get_fscore(self):
        if self.model == None:
            raise Exception('Please fit the data by using model before predict') 
        try:
            df = pd.DataFrame(pd.Series(self.model.get_booster().get_fscore())).reset_index(drop=False)
        except:
            df = pd.DataFrame(pd.Series(self.model.get_fscore())).reset_index(drop=False)
        df.columns = ['feature_name', 'feature_importance']
        return df.sort_values('feature_importance', ascending=False)


    def find_best_params(self, x, y, param_grid):
        '''
        根据x y 和model，找到最佳的参数，并返回最佳的model
        clf 是xgboost的sklearn包的模型
        '''
        if self.model == None:
            # self.train(x, y, x, y, is_eval=False)
            self.init_model()

        gsearch = GridSearchCV(
                    estimator=self.model, 
                    param_grid=param_grid, 
                    n_jobs=4,
                    iid=False, 
                    cv=5)
        gsearch.fit(x, y)
        return gsearch.best_estimator_


    def get_stacking(self, x, y, x_test, y_test, n_folds=5, random_state=2017):
        '''
        Args:
            clf: 模型
            x, y: 训练data和label，用于K-folds stacking
            x_test, y_test: valid数据，非必须，如没有的可直接用x和y

        Returns:
            oof_train: K-folds stacking出来的K份test数据，最终合并一起
            oof_test: K-folds stacking出来K份预测结果的均值
        '''
        """ K-fold stacking """

        from sklearn.model_selection import KFold
        cols = x.columns.tolist()
        x, y = np.array(x), np.array(y)
        # x_test, y_test = np.array(x_test), np.array(y_test)
        num_train, num_test = x.shape[0], x_test.shape[0]
        oof_train = np.zeros((num_train,)) 
        oof_test  = np.zeros((num_test, ))
        oof_test_all_fold = np.zeros((num_test, n_folds))

        KF = KFold(n_splits=n_folds, random_state=random_state)
        for i, (tra_idx, val_idx) in enumerate(KF.split(x, y=y)):
            print(f'{i} fold - get_stacking, train {len(tra_idx)}, val {len(val_idx)}\n')

            x_tra, y_tra = x[tra_idx], y[tra_idx]
            x_val, y_val = x[val_idx], y[val_idx]
            x_tra = pd.DataFrame(np.array(x_tra), columns=cols)
            x_val = pd.DataFrame(np.array(x_val), columns=cols)

            self.train(x_tra, y_tra, x_val, y_val)
            oof_train[val_idx] = self.predict(x_val)
            oof_test_all_fold[:, i] = self.predict(x_test)
        oof_test = np.mean(oof_test_all_fold, axis=1)
        return oof_train, oof_test


    def cross_validation(self, x, y, x_test, y_test, n_folds=5, random_state=2017):
        """ K-fold cross_validation """
        x, y = np.array(x), np.array(y)
        x_test, y_test = np.array(x_test), np.array(y_test)
        # print('>>> mean(y) =', np.mean(y))
        # print('>>> mean(y_test) =', np.mean(y_test))

        aucs_tra = []
        aucs_val = []
        aucs_tes = []

        # KF = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        SKF = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        # for i, (tra_idx, val_idx) in enumerate(KF.split(x)):
        for i, (tra_idx, val_idx) in enumerate(SKF.split(x, y)):
            print(f'>>> {i} fold, train {len(tra_idx)}, val {len(val_idx)}')
            
            x_tra, y_tra = x[tra_idx], y[tra_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            # 需要计算 train， validation 和 test三个的AUC
            self.train(x_tra, y_tra, x_val, y_val)
            
            y_pred = self.predict(x_tra)
            auc = metrics.roc_auc_score(y_tra,  y_pred)
            aucs_tra.append(auc)

            y_pred = self.predict(x_val)
            auc = metrics.roc_auc_score(y_val,  y_pred)
            aucs_val.append(auc)

            y_pred = self.predict(x_test)
            auc = metrics.roc_auc_score(y_test,  y_pred)
            aucs_tes.append(auc)
            print()

        return pd.DataFrame([aucs_tra, aucs_val, aucs_tes], index=['tra_auc', 'val_auc', 'test_auc']).T