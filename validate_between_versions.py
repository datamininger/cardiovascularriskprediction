from utils import CombineFeature, read_log, read_model_file
import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb

def validate_between_versions(log1, v2_name):
    """
    假设使用版本 v1_name 本地交叉验证过一次，其log为log1， 利用这个Log读取模型，预测v2_name特征，并计算分数
    :param log1:
    :param v2_name:
    :return:
    """
    # 模型等相关数据准备
    log = read_log(log1)
    feature_names = log['config']['feature_names']
    config_name = log['config']['config_name']
    model_name = log['config']['model']['model_name']
    best_iteration = log['result']['best_iteration']
    num_model = len(best_iteration)
    models = read_model_file(config_name, model_name, num_model)

    # 读取v2_name 版本的特征
    Xtrain, Ytrain, Xtest = CombineFeature(feature_names=feature_names, versions=[v2_name], test=False)
    if model_name == 'xgboost':
        Xtrain = xgb.DMatrix(Xtrain[feature_names].values, feature_names=feature_names)

    # 预测
    submission_list = []
    for model, best_iter in zip(models, best_iteration):
        if model_name == 'xgboost':
            submission_list.append(model.predict(Xtrain, ntree_limit=best_iter))
        elif model_name == 'lightgbm':
            submission_list.append(model.predict(Xtrain[feature_names].values, num_iteration=best_iter))
    y_pred = np.mean(submission_list, axis=0)
    auc = roc_auc_score(Ytrain['label'], y_pred)
    print('auc is {}'.format(auc))
    return auc

