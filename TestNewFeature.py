# coding: utf-8

from utils import read_feature_names_from_hist, read_log, get_path_feature, IsDifferentDistribution
from Code.Config.run_config import run_config
from Code.Config.gen_config import gen_config
import pandas as pd


def combine_feature_names(log_names, feat_hist_file_name, batch_names):
    """
    :param log_names: list, 日志名列表
    :param feat_hist_file_name: str, 特征生成历史文件名
    :param batch_names: list, 批次名列表
    :return:
    """
    old_feature_names = []
    # 基础特征
    for log_name in log_names:
        log = read_log(log_name)
        feats = log['config']['feature_names']
        old_feature_names += feats
    # 添加需要被测试的新特征
    new_feature_names = read_feature_names_from_hist(feat_hist_file_name, batch_names)
    return old_feature_names, new_feature_names


def select_feature(log_name, n):
    """
    从日志中挑选前n个特征
    :param log_name:
    :param n:
    :return:
    """
    feature_names = []
    log = read_log(log_name)
    fold_results = log['result']['fold_results']
    zero_feature_names = []
    for fold_result in fold_results:
        feats = [feat_tuple[0] for feat_tuple in sorted(fold_result['feature_importance_dict'].items(), key=lambda item: item[1]) if feat_tuple[1] != 0]
        feature_names += feats[-n:]
        zero_feature_names += [feat_tuple[0] for feat_tuple in sorted(fold_result['feature_importance_dict'].items(), key=lambda item: item[1]) if
                               feat_tuple[1] == 0]

    feature_names = list(set(feature_names))
    non_zero_feature_names = []
    for feat in feature_names:
        if feat not in zero_feature_names:
            non_zero_feature_names.append(feat)
    print('The number of feature_names is ', len(feature_names))
    return non_zero_feature_names


def main():
    """
    测试特征使用
    :return:
    """
    # 0 计算新老特征
    log_names = [20]
    feat_hist_file_name = 'stats_vector_return_value_without_thread'
    batch_names = [str(i) for i in range(10, 16)]
    old_feature_names, new_feature_names = combine_feature_names(log_names, feat_hist_file_name, batch_names)
    feature_names = old_feature_names + new_feature_names

    # feature_names = select_feature(21, 1200)
    # 1 生成配置
    config_name = 22
    versions = ['v1']
    model_name = 'lightgbm'
    save_log = True
    config = gen_config(versions=versions, config_name=config_name, feature_names=feature_names, model_name=model_name, save_log=save_log)
    # 2 跑一次配置
    run_config(config)


main()
