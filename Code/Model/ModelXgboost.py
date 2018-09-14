# coding utf-8

import xgboost as xgb
import numpy as np
import copy


class ModelXgboost:
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
        # xgboost的模型参数，训练参数
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
            # 2 转化格式
            xgb_train = xgb.DMatrix(data=X_train[feature_names].values, label=y_train['label'].values, feature_names=feature_names)

            xgb_valid = xgb.DMatrix(data=X_valid[feature_names].values, label=y_valid['label'].values, feature_names=feature_names)
            # 3 训练,
            eval_result = {}
            booster_offline = xgb.train(params=self.model_config['model_params'],
                                        dtrain=xgb_train,
                                        evals=[(xgb_train, 'train'), (xgb_valid, 'valid')],
                                        evals_result=eval_result,
                                        **self.model_config['train_params']
                                        )
            # 4 微微的收集工作
            self.booster_offline_list.append(booster_offline)
            self.eval_result_list.append(eval_result)
        return self.booster_offline_list, self.eval_result_list

    def online_predict(self, oof=True, best_iteration=None):
        """
        :param oof: bool, True 那么使用折外的数据训练模型分别预测取平均
        :param best_iteration: int, 默认值为None，使用train_params中的迭次次数
        :return:
        """
        # 1 转化格式
        feature_names = list(self.Xtrain.columns)
        xgb_test = xgb.DMatrix(self.Xtest[feature_names].values, feature_names=feature_names)
        # 2 是否oof
        if oof:
            print('online predict by oof ...')
            submissions = []
            for booster in self.booster_offline_list:
                submissions.append(booster.predict(xgb_test, ntree_limit=booster.best_iteration))
            self.submission_online = np.mean(submissions, axis=0)
        else:
            # 训练参数
            train_params = copy.deepcopy(self.model_config['train_params'])
            del train_params["early_stopping_rounds"]
            if best_iteration:
                train_params["num_boost_round"] = best_iteration
            # 开始训练
            print('online predict by all train data ...')
            xgb_train = xgb.DMatrix(self.Xtrain[feature_names].values, self.Ytrain['label'].values, feature_names=feature_names)
            booster_online = xgb.train(params=self.model_config['model_params'],
                                       dtrain=xgb_train,
                                       **train_params)
            # 预测
            self.submission_online = booster_online.predict(xgb_test)
        return self.submission_online
