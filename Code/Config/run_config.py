# coding: utf-8
from utils import CombineFeature
from Code.CrossValidation.CrossValidation import gen_train_validate_index
from Code.Model.InitModel import InitModel
from Code.Model.PostProcessing import PostProcessing
import pandas as pd


def run_config(config):
    """
    运行一次配置
    :param config: dict, 配置字典
    :return:
    """
    # 0 取出接下来用到的部分变量
    config_name = config['config_name']
    versions = config['versions']
    feature_names = config['feature_names']
    model_config = config['model']
    model_name = model_config['model_name']
    validate_params = config['validate_params']
    oof = config['oof']
    submission_online = []

    # 1.1 根据配置合并多个版本的特征,
    Xtrain, Ytrain, Xtest = CombineFeature(feature_names=feature_names, versions=versions, test=config['test'], test_version=config['test_version'])
    # 1.2 划分线下训练验证集合
    train_validate_index = gen_train_validate_index(Xtrain=Xtrain, Ytrain=Ytrain, validate_params=validate_params)
    Xtrain = Xtrain.drop(['ID', 'TimePoint', 'version'], axis=1)
    Ytrain = Ytrain.drop(['ID', 'TimePoint', 'version'], axis=1)
    if config['test']:
        test_labels = Xtest[['ID', 'TimePoint', 'version']]
        Xtest = Xtest.drop(['ID', 'TimePoint', 'version'], axis=1)
    # 2 初始化模型
    model = InitModel(Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, model_config=model_config)
    # 3 线下验证
    booster_offline_list, eval_result_list = model.offline_validate(train_validate_index)
    # 4 线上预测
    if config['test']:
        submission_online = model.online_predict(oof=oof, best_iteration=None)
        submission_online = pd.concat([test_labels, pd.DataFrame(data=submission_online,columns=['prob'])], axis=1)
    # 5 后处理
    ps = PostProcessing(booster_offline_list, eval_result_list, submission_online)
    ps.gen_validate_results(model_name, validate_params['norm_feat_imp'], feature_names)
    ps.save_validate_model(config_name, model_name)
    if config['save_log']:
        ps.save_log(config)
    if config['test']:
        ps.save_submission(config_name)
    # 6 返回模型
    return model
