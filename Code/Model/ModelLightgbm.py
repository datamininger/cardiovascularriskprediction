# coding utf-8

import lightgbm as lgb
import numpy as np
import copy
from lightgbm import LGBMClassifier


class ModelLightgbm:
    """
    功能一: 线下验证，每折训练模型和验证结果分别保存在一个列表中
    功能二: 线上预测，可决定是否使用oof,重新设置迭代次数
    """

    def __init__(self, Xtrain, Ytrain, Xtest, model_config):
        """
        :param Xtrain: pd.DataFrame
        :param Ytrain: pd.DataFrame
        :param Xtest: pd.DataFrame
        :param model_config: dict
        """
        # 训练特征，训练标签，测试特征
        self.Xtrain, self.Ytrain, self.Xtest = Xtrain, Ytrain, Xtest
        # xgboost 配置参数模型包括名字包模型参数，训练参数
        self.model_config = model_config
        # 保持模型用的列表
        self.booster_offline_list = []
        # 保存验证结果的列表
        self.eval_result_list = []
        # 预测值
        self.submission_online = None

    def offline_validate(self, train_validate_index):
        """
        :param train_validate_index: like list, [train_index, validate_index]
        :return: self.booster_offline_list, self.eval_result_list
        """
        feature_names = list(self.Xtrain.columns)

        for train_index, valid_index in train_validate_index:
            # 1 分割数据
            X_train, X_valid, y_train, y_valid = self.Xtrain.iloc[train_index], self.Xtrain.iloc[valid_index], \
                                                 self.Ytrain.iloc[train_index], self.Ytrain.iloc[valid_index]
            booster_offline = LGBMClassifier(**self.model_config['model_params'])

            booster_offline = booster_offline.fit(X=X_train[feature_names].values,
                                                  y=y_train['label'].values,
                                                  eval_set=[(X_train[feature_names].values, y_train['label'].values), (X_valid[feature_names].values, y_valid['label'].values)],
                                                  eval_names=['train', 'valid'],
                                                  feature_name=feature_names,
                                                  **self.model_config['train_params'])
            self.booster_offline_list.append(booster_offline)
            self.eval_result_list.append(booster_offline.evals_result_)
        return self.booster_offline_list, self.eval_result_list

    def online_predict(self, oof=True, best_iteration=None):
        """
        :param oof: bool, True 那么使用折外的数据训练模型分别预测取平均
        :param best_iteration: int, 默认值为None，使用train_params中的迭次次数
        :return:
        """
        feature_names = list(self.Xtrain.columns)
        if oof:
            submissions = []
            for booster in self.booster_offline_list:
                submissions.append(booster.predict(self.Xtest[feature_names].values, num_iteration=booster.best_iteration))
            self.submission_online = np.mean(submissions, axis=0)
        else:
            train_params = copy.deepcopy(self.model_config['train_params'])
            del train_params["early_stopping_rounds"]
            if best_iteration:
                train_params["num_boost_round"] = best_iteration
            lgb_train = lgb.Dataset(data=self.Xtrain[feature_names].values,
                                    label=self.Ytrain['label'].values)
            booster_online = lgb.train(params=self.model_config['model_params'],
                                       train_set=lgb_train,
                                       feature_name=feature_names,
                                       **train_params)
            self.submission_online = booster_online.predict(self.Xtest[feature_names].values)
        return self.submission_online
