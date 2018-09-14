# coding: utf-8
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import pdb
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import os

# ------------------------------------------------------------------------------------------------------------------------------------------------
num_train = 0
num_test = 0


# ------------------------------------------------------------------------------------------------------------------------------------------------


# 路径配置

def get_path():
    return '/root/code/xiao/cardiovascularriskprediction/'


def get_path_log():
    return get_path() + 'Output/Log/'


def get_path_submission():
    return get_path() + 'Output/Submission/'


def get_path_feature():
    return get_path() + 'Data/Feature/'


def get_path_raw_data():
    return get_path() + 'Data/RawData/'


def get_path_model_file():
    return get_path() + 'Output/ModelFile/'


def get_path_feat_hist():
    return get_path() + 'Output/FeatGenHist/'


def get_path_labels():
    return get_path() + 'Output/Labels/'


# ------------------------------------------------------------------------------------------------------------------------------------------------
# 读取历史数据
def ReadHistData(version, info, kind):
    """
    :param kind: 哪种历史数据
    :return:
    """
    assert kind in ['train', 'test']
    path_raw_data = get_path_raw_data()
    history_data = pd.read_pickle(path_raw_data + '{}_{}_{}.pkl'.format(version, info, kind))
    return history_data


# ------------------------------------------------------------------------------------------------------------------------------------------------
# 读取标的

def ReadLabelsData(version, kind):
    """
    :param version: str, 标的版本
    :param kind: str, train or test
    :return:
    """
    labels = pd.read_pickle(get_path_labels() + version + '_labels_' + kind + '.pkl')
    return labels


# 保存标的

def SaveLabelsData(df_labels, params_dict, kind='train'):
    version = params_dict['version']
    df_labels.to_pickle(get_path_labels() + version + '_labels_{}.pkl'.format(kind))
    labels_hist = json.load(open(get_path_labels() + 'LabelsHist.json'))
    labels_hist[version] = params_dict
    json.dump(labels_hist, open(get_path_labels() + 'LabelsHist.json', 'w'), indent=2)


# ------------------------------------------------------------------------------------------------------------------------------------------------
# 读取模型文件

def read_model_file(config_name, model_name, n):
    """
    :param config_name: 哪个配置保存的模型
    :param model_name: 模型名
    :param n: 模型个个数
    :return:
    """
    models = []
    if model_name == 'xgboost':
        for i in range(n):
            model = xgb.Booster(model_file=get_path_model_file() + '{}_{}_{}.m'.format(config_name, model_name, i))
            models.append(model)
    elif model_name == 'lightgbm':
        for i in range(n):
            model = lgb.Booster(model_file=get_path_model_file() + '{}_{}_{}.m'.format(config_name, model_name, i))
            models.append(model)
    return models


# ------------------------------------------------------------------------------------------------------------------------------------------------

# 特征是否存在

def ExistFeature(feat, version, kind):
    """
    :param feat: str, 特征名
    :param version: str, 版本号
    :param kind: str, train or test
    :return:
    """
    path_feature = get_path_feature()
    exists = os.path.exists(path_feature + version + '_' + feat + '_' + kind + '.pkl')
    return exists


# 保存合并特征
def SaveFeature(feat_df, feature_name, version, kind):
    """
    保存特征, 特征文件名为 FeatureName_{train or test}.pkl
    :param feat_df: pd.DataFrame
    :param feature_name: str
    :param version: str
    :param kind: str
    :return:
    """
    assert kind in ['train', 'test']
    path_feature = get_path_feature()
    feat_df[['ID', 'TimePoint', 'version', feature_name]].to_pickle(path_feature + version + '_' + feature_name + '_{}.pkl'.format(kind))
    return


def CombineFeature(feature_names, versions, test, test_version):
    """
    利用特征文件名读取特征拼接成 DataFrame
    每个特征文件都是DataFrame,列名都是ID feature_name
    一个文件只有一个特征, 使用merge合并所有文件，最后再对对齐Ytrain
    :param feature_names: list, list of the feature file name
    :param  versions: list, 训练特征的所有版本号
    :return: pd.DataFrame, pd.DataFrame, pd.DataFrame, Xtrain, Ytrain, Xtest, 均含 ID
    """
    path_feature = get_path_feature()
    # 文件格式
    file_format = '.pkl'
    # 空DataFrame
    Xtrain, Xtest = pd.DataFrame({}), pd.DataFrame({})
    # 1 读取训练标签
    print('Reading Ytrain...')
    # 2 两个循环分别读取版本和特征
    # 训练特征
    Xtrain_list = []
    for version in versions:
        Xtrain_version = pd.read_pickle(path_feature + version + '_' + feature_names[0] + '_train' + file_format)
        print('     Reading {} {} '.format(version, feature_names[0]))
        for feature_name in feature_names[1:]:
            print('     Reading {} {} '.format(version, feature_name))
            train_feature = pd.read_pickle(path_feature + version + '_' + feature_name + '_train' + file_format)
            Xtrain_version = Xtrain_version.merge(train_feature, on=['ID', 'TimePoint', 'version'], how='left')
        Xtrain_list.append(Xtrain_version)
    Xtrain = pd.concat(Xtrain_list, axis=0, ignore_index=True)
    # 训练标签
    Ytrain_list = []
    for version in versions:
        Ytrain_version = ReadLabelsData(version=version, kind='train')
        Ytrain_list.append(Ytrain_version)
    Ytrain = pd.concat(Ytrain_list, axis=0, ignore_index=True)
    # 测试特征
    if test:
        Xtest = pd.read_pickle(path_feature + test_version + '_' + feature_names[0] + '_test' + file_format)
        for feature_name in feature_names[1:]:
            test_feature = pd.read_pickle(path_feature + test_version + '_' + feature_name + '_test' + file_format)
            Xtest = Xtest.merge(test_feature, on=['ID', 'TimePoint', 'version'], how='left')

    # 再次对齐Ytrain
    Xtrain = Xtrain.merge(Ytrain, on=['ID', 'TimePoint', 'version'], how='left')
    Ytrain = Xtrain[['ID', 'TimePoint', 'version', 'label']]
    Xtrain = Xtrain.drop(['label'], axis=1)
    print('Finished Combine Feature')
    return Xtrain, Ytrain, Xtest


# ------------------------------------------------------------------------------------------------------------------------------------------------
# 日志 特征历史文件处理
def read_feat_hist(feat_hist_file_name):
    """
    读取特征生成历史
    :param feat_hist_file_name: str
    :return:
    """
    file_path = get_path_feat_hist() + feat_hist_file_name + '.json'
    feat_hist = json.load(open(file_path))
    return feat_hist


def read_feature_names_from_hist(feat_hist_file_name, batch_names):
    """
    读取特征生成文件中的特征名字
    :param feat_hist_file_name: str,
    :param batch_names: list
    :return:
    """
    feature_names = []
    feat_hist = read_feat_hist(feat_hist_file_name)
    for batch_name in batch_names:
        feature_names += feat_hist[batch_name]
    return feature_names


def read_log(log_name):
    """
    读取日志
    :param log_name: str
    :return: dict
    """
    path_log = get_path_log() + '{}.json'.format(log_name)
    log = json.load(open(path_log))
    return log


def SaveFeatHist(feat_hist_file_name, i, feat_names):
    """
    保存特征生成历史
    :param feat_hist_file_name: str
    :param i: int or str, batch_name
    :param feat_names: list
    :return:
    """
    feat_hist = read_feat_hist(feat_hist_file_name)
    if i in feat_hist:
        print('The batch has been computed...')
    else:
        feat_hist[i] = feat_names
    json.dump(feat_hist, open(get_path_feat_hist() + feat_hist_file_name + '.json', 'w'), indent=2)


# ------------------------------------------------------------------------------------------------------------------------------------------------
# 结果 后处理辅助函数


def compute_validate_result_xgboost(booster_offline, eval_result, norm):
    """
    计算最优迭代次数， 线下分数， 特征重要性
    :param booster_offline: booster,
    :param eval_result:dict,
    :return: dict
    """
    best_iteration = booster_offline.best_iteration
    score_offline = eval_result['valid']['auc'][best_iteration]
    fscore_dict = booster_offline.get_fscore()
    fscore_series = pd.Series(fscore_dict)
    if norm:
        fscore_series = fscore_series / fscore_series.sum()

    validate_result = {'best_iteration': best_iteration,
                       'score_offline': score_offline,
                       'feature_importance_dict': fscore_series.to_dict()}
    return validate_result


def compute_validate_result_lightgbm(booster_offline, eval_result, norm, feature_names):
    """
     计算最优迭代次数， 线下分数， 特征重要性
    :param booster_offline:
    :param eval_result:
    :param feature_names:
    :param norm:bool
    :return:
    """
    best_iteration = booster_offline.best_iteration_
    score_offline = eval_result['valid']['auc'][best_iteration]
    feature_importance_list = booster_offline.feature_importances_
    fscore_dict = {feat_name: feat_imp for feat_name, feat_imp in zip(feature_names, feature_importance_list)}
    fscore_series = pd.Series(fscore_dict)
    if norm:
        fscore_series = fscore_series / fscore_series.sum()
    validate_result = {'best_iteration': best_iteration,
                       'score_offline': score_offline,
                       'feature_importance_dict': fscore_series.to_dict()}
    return validate_result


def ensemble_submission(submissions, weights, ensemble_method):
    """
    :param submissions: list,
    :param weights: list
    :param ensemble_method: str
    :return: list
    """
    assert ensemble_method in ['mean', 'rank_mean', 'mean_weight', 'rank_mean_weight']
    print('ensemble method is {}'.format(ensemble_method))
    if ensemble_method == 'mean':
        submission_online = np.mean(submissions, axis=0)
    if ensemble_method == 'rank_mean':
        submission_set = []
        for submission in submissions:
            submission_set.append(pd.Series(submission).rank().values)
        submission_online = np.mean(submission_set, axis=0)
    if ensemble_method == 'mean_weight':
        submission_set = []
        for submission, weight in zip(submissions, weights):
            submission_set.append(weight * pd.Series(submission).values)
        submission_online = np.sum(submission_set, axis=0) / np.sum(weights)
    if ensemble_method == 'rank_mean_weight':
        submission_set = []
        for submission, weight in zip(submissions, weights):
            submission_set.append(weight * pd.Series(submission).rank().values)
        submission_online = np.sum(submission_set, axis=0) / np.sum(weights)
    return submission_online


# ------------------------------------------------------------------------------------------------------------------------------------------------
#

def AddBlackFeature(feature_names):
    """
    :param feature_names: list, 新添加的黑特征
    :return:
    """
    # 2 读取
    path = get_path()
    with open(path + 'BlackFeature.json') as f:
        black_dict = json.load(f)

    black_list = black_dict['black_list']
    for feature in feature_names:
        black_list.append(feature)
    black_dict['black_list'] = black_list
    json.dump(black_dict, open(path + 'BlackFeature.json', 'w'), indent=2)


def IsDifferentDistribution(feature, bins=100):
    """
    画分布图，人工判断是否具有不同分布，并输入数字
    :param feature:
    :return:
    """
    print('Plot the distribution of {}'.format(feature))
    Xtrain, Ytrain, Xtest = CombineFeature([feature])
    Xtrain['label'] = Ytrain['label'].values
    sns.distplot(Xtrain[feature].values, bins=bins, hist=True, kde=False, norm_hist=True, label='train')
    sns.distplot(Xtest[feature].values, bins=bins, hist=True, kde=False, norm_hist=True, label='test')
    plt.title(feature)
    plt.legend()
    plt.show()

    sns.violinplot(x='label', y=feature, data=Xtrain)
    plt.title(feature)
    plt.show()

    IsDifferent = int(input('IsDifferent: '))
    if IsDifferent > 0:
        print('{} is a  black feature'.format(feature))
        return True
    else:
        print('{} is not  a black feature'.format(feature))
        return False


# ------------------------------------------------------------------------------------------------------------------------------------------------
# 用时间和ID过滤数据
def filter_hist_data(label, followup):
    """
    :param label:
    :param followup:
    :return:
    """
    mask1 = followup['ID'] == label['ID']
    mask2 = followup['FOLLOWUP_DATE'] <= label['TimePoint']
    mask = mask1 & mask2
    df_person = followup[mask]
    df_person = df_person.reset_index(drop=True)
    return df_person


# ------------------------------------------------------------------------------------------------------------------------------------------------
# 统计随访表中人员各种指标

def compute_stats_value_dict(df_person, values, stats_name, agg):
    """
    :param df_person: pd.DataFrame,
    :param feat: str, 统计的指标
    :param stats_name:str, 统计名
    :return:
    """
    assert stats_name in ['mean', 'max', 'min', 'range', 'std']
    if agg == 'mean':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].mean().reset_index()
    if agg == 'max':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].max().reset_index()
    if agg == 'min':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].min().reset_index()

    stats_dict = {}
    if stats_name == 'mean':
        stats_series = df_person[values].mean(axis=0)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'max':
        stats_series = df_person[values].max(axis=0)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'min':
        stats_series = df_person[values].min(axis=0)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'range':
        stats_series = df_person[values].max(axis=0) - df_person[values].min(axis=0)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'std':
        stats_series = df_person[values].std(axis=0)
        stats_dict = stats_series.to_dict()
    return stats_dict


# ------------------------------------------------------------------------------------------------------------------------------------------------
# 统计随访表中人员时间特征
# 速度
def speed(col, time_diff):
    speed = col.diff() / time_diff
    return speed


# 加速度
def acc(col, time_diff):
    speed_ = speed(col, time_diff)
    acc = speed(speed_, time_diff)
    return acc


def compute_time_feature_dict(df_person, values, stats_name, agg):
    """
    :param df_person:
    :param values:
    :param stats_name:
    :param agg:
    :return:
    """
    stats_names = ['speed_mean', 'speed_max', 'speed_min', 'speed_range', 'speed_std',  # 速度
                   'acc_mean', 'acc_max', 'acc_min', 'acc_range', 'acc_std',  # 加速度
                   'value_last', 'speed_last', 'acc_last',  # 最后一次
                   'rolling_max'
                   ]
    assert stats_name in stats_names
    # 同日期聚合一次
    if agg == 'mean':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].mean().reset_index()
    elif agg == 'max':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].max().reset_index()
    elif agg == 'min':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].min().reset_index()
    else:
        assert False
    df_person = df_person.sort_values(by=['FOLLOWUP_DATE']).reset_index(drop=True)
    # 使用聚合的df_person开始统计
    # -----------------------------------------------------------------------------------------------------------
    # 速度

    if stats_name == 'speed_mean':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_series = df_person[values].apply(lambda col: speed(col, time_diff).mean(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'speed_max':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_series = df_person[values].apply(lambda col: speed(col, time_diff).max(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'speed_min':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_series = df_person[values].apply(lambda col: speed(col, time_diff).min(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'speed_range':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_series_max = df_person[values].apply(lambda col: speed(col, time_diff).max(), axis=0)
        stats_series_min = df_person[values].apply(lambda col: speed(col, time_diff).min(), axis=0)
        stats_series = stats_series_max - stats_series_min
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'speed_std':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_series = df_person[values].apply(lambda col: speed(col, time_diff).std(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()

    # ----------------------------------------------------------------------------------------------------------
    # 加速度

    elif stats_name == 'acc_mean':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_series = df_person[values].apply(lambda col: acc(col, time_diff).mean(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'acc_max':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_series = df_person[values].apply(lambda col: acc(col, time_diff).max(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'acc_min':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_series = df_person[values].apply(lambda col: acc(col, time_diff).min(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'acc_range':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_series_max = df_person[values].apply(lambda col: acc(col, time_diff).max(), axis=0)
        stats_series_min = df_person[values].apply(lambda col: acc(col, time_diff).min(), axis=0)
        stats_series = stats_series_max - stats_series_min
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'acc_std':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_series = df_person[values].apply(lambda col: acc(col, time_diff).std(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()

    # ------------------------------------------------------------------------------------------------------------
    # 最后一次时间的特征

    elif stats_name == 'value_last':
        stats_series = df_person[values].apply(lambda col: col.values[-1], axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
        for key in stats_dict.keys():
            if stats_dict[key] == {}:
                stats_dict[key] = -99999
    elif stats_name == 'speed_last':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_series = df_person[values].apply(lambda col: speed(col, time_diff).values[-1], axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
        for key in stats_dict.keys():
            if stats_dict[key] == {}:
                stats_dict[key] = -99999
    elif stats_name == 'acc_last':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_series = df_person[values].apply(lambda col: acc(col, time_diff).values[-1], axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
        for key in stats_dict.keys():
            if stats_dict[key] == {}:
                stats_dict[key] = -99999
    else:
        assert False

    # -----------------------------------------------------------------------------------------------------------
    # 滑窗提取局部特征

    return stats_dict


# -----------------------------------------------------------------------------------------------------------------------------
# 测压活动时间

def compute_action_time_feature(df_person, stats_name):
    """
    :param df_person:
    :param stats_name:
    :return:
    """
    stats_names = ['time_diff_mean', 'time_diff_max', 'time_diff_min', 'time_diff_range', 'time_diff_std',
                   'time_diff_diff_mean', 'time_diff_diff_max', 'time_diff_diff_min', 'time_diff_diff_range', 'time_diff_diff_std']
    df_person = df_person.sort_values(by=['FOLLOWUP_DATE'])

    # --------------------------------------------------------------------
    # 一阶
    if stats_name == 'time_diff_mean':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_value = time_diff.mean()
        if pd.isna(stats_value):
            stats_value = (-9999)
    elif stats_name == 'time_diff_max':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_value = time_diff.max()
        if pd.isna(stats_value):
            stats_value = (-9999)
    elif stats_name == 'time_diff_min':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_value = time_diff.min()
        if pd.isna(stats_value):
            stats_value = (-9999)
    elif stats_name == 'time_diff_range':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_value_max = time_diff.max()
        stats_value_min = time_diff.min()
        stats_value = stats_value_max - stats_value_min
        if pd.isna(stats_value):
            stats_value = (-9999)
    elif stats_name == 'time_diff_std':
        time_diff = df_person['FOLLOWUP_DATE'].diff().dt.days
        stats_value = time_diff.std()
        if pd.isna(stats_value):
            stats_value = (-9999)
    # -----------------------------------------------------------------------
    # 二阶
    elif stats_name == 'time_diff_diff_mean':
        time_diff = df_person['FOLLOWUP_DATE'].diff().diff().dt.days
        stats_value = time_diff.mean()
        if pd.isna(stats_value):
            stats_value = (-9999)
    elif stats_name == 'time_diff_diff_max':
        time_diff = df_person['FOLLOWUP_DATE'].diff().diff().dt.days
        stats_value = time_diff.max()
        if pd.isna(stats_value):
            stats_value = (-9999)
    elif stats_name == 'time_diff_diff_min':
        time_diff = df_person['FOLLOWUP_DATE'].diff().diff().dt.days
        stats_value = time_diff.min()
        if pd.isna(stats_value):
            stats_value = (-9999)
    elif stats_name == 'time_diff_diff_range':
        time_diff = df_person['FOLLOWUP_DATE'].diff().diff().dt.days
        stats_value_max = time_diff.max()
        stats_value_min = time_diff.min()
        stats_value = stats_value_max - stats_value_min
        if pd.isna(stats_value):
            stats_value = (-9999)
    elif stats_name == 'time_diff_diff_std':
        time_diff = df_person['FOLLOWUP_DATE'].diff().diff().dt.days
        stats_value = time_diff.std()
        if pd.isna(stats_value):
            stats_value = (-9999)
    else:
        assert False
    return stats_value


# ---------------------------------------------------------------------------------------------------------------------------------------
# 缺失值特征

def compute_stats_missing_dict(df_person, values, stats_name):
    """
    :param df_person:
    :param stats_name:
    :return:
    """
    stats_names = ['num_mean_na_single', 'num_mean_na_all',
                   'num_max_na_single', 'num_max_na_all',
                   'num_min_na_single', 'num_min_na_all',
                   'num_range_na_single', 'num_range_na_all',
                   'num_std_na_single', 'num_std_na_all',

                   'num_diff_mean_na_single', 'num_diff_mean_na_all',
                   'num_diff_max_na_single', 'num_diff_max_na_all',
                   'num_diff_min_na_single', 'num_diff_min_na_all',
                   'num_diff_range_na_single', 'num_diff_range_na_all',
                   'num_diff_std_na_single', 'num_diff_std_na_all',
                   ]

    # ----------------------------------------------------------------------------------------------------------------------------
    def excist_na(col):
        x = col.isnull().sum()
        if x > 0:
            return 1
        else:
            return 0

    if stats_name == 'num_mean_na_single':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: excist_na(col)).reset_index()
        stats_series = df_person.mean(axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_max_na_single':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: excist_na(col)).reset_index()
        stats_series = df_person.max(axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_min_na_single':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: excist_na(col)).reset_index()
        stats_series = df_person.min(axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_range_na_single':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: excist_na(col)).reset_index()
        stats_series_max = df_person.max(axis=0)
        stats_series_min = df_person.min(axis=0)
        stats_series = stats_series_max - stats_series_min
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_std_na_single':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: excist_na(col)).reset_index()
        stats_series = df_person.std(axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()

    # --------------------------------------------------------------------------------------------------------------------------
    elif stats_name == 'num_diff_mean_na_single':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: excist_na(col)).reset_index()
        stats_series = df_person.apply(lambda col: col.diff().mean(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_diff_max_na_single':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: excist_na(col)).reset_index()
        stats_series = df_person.apply(lambda col: col.diff().max(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_diff_min_na_single':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: excist_na(col)).reset_index()
        stats_series = df_person.apply(lambda col: col.diff().min(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_diff_range_na_single':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: excist_na(col)).reset_index()
        stats_series = df_person.apply(lambda col: col.diff().max() - col.diff().min(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_diff_std_na_single':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: excist_na(col)).reset_index()
        stats_series = df_person.apply(lambda col: col.diff().std(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()

    # ----------------------------------------------------------------------------------------------------------------------------
    elif stats_name == 'num_mean_na_all':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: col.isnull().sum()).reset_index()
        stats_series = df_person.mean(axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_max_na_all':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: col.isnull().sum()).reset_index()
        stats_series = df_person.max(axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_min_na_all':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: col.isnull().sum()).reset_index()
        stats_series = df_person.min(axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_range_na_all':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: col.isnull().sum()).reset_index()
        stats_series_max = df_person.max(axis=0)
        stats_series_min = df_person.min(axis=0)
        stats_series = stats_series_max - stats_series_min
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_std_na_all':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: col.isnull().sum()).reset_index()
        stats_series = df_person.std(axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()

    # --------------------------------------------------------------------------------------------------------------------------
    elif stats_name == 'num_diff_mean_na_all':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: col.isnull().sum()).reset_index()
        stats_series = df_person.apply(lambda col: col.diff().mean(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_diff_max_na_all':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: col.isnull().sum()).reset_index()
        stats_series = df_person.apply(lambda col: col.diff().max(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_diff_min_na_all':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: col.isnull().sum()).reset_index()
        stats_series = df_person.apply(lambda col: col.diff().min(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_diff_range_na_all':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: col.isnull().sum()).reset_index()
        stats_series = df_person.apply(lambda col: col.diff().max() - col.diff().min(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    elif stats_name == 'num_diff_std_na_all':
        df_person = df_person.groupby('FOLLOWUP_DATE')[values].apply(lambda col: col.isnull().sum()).reset_index()
        stats_series = df_person.apply(lambda col: col.diff().std(), axis=0)
        stats_series = stats_series.fillna(-9999)
        stats_dict = stats_series.to_dict()
    return stats_dict
