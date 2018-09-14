import xgboost as xgb
from utils import get_path_feature, SaveFeature
import pandas as pd
import os
import multiprocessing


def fill_na(value, train_version='v2', test_version='v2_1'):
    """
    :param value: str, 被预测的指标
    :param train_version: list, 训练回归模型的特征版本
    :param test_version: str, 被预测填充的测试版本
    :return:
    """
    # 1 不同的测试子集使用不同的训练特征来预测
    if test_version == 'v2_1':
        feature_names = ['SEX_CODE', 'age', 'confirm_age', 'time_diff_confirm_2TimePoint']
    elif test_version in ['v2_2', 'v2_3', 'v2_4']:
        feature_names = ['SEX_CODE', 'age', 'confirm_age', 'time_diff_confirm_2TimePoint',
                         'first_followup_age', 'first_followup_time_diff_confirm_time', 'TimePoint_diff_first_followup_time']
    else:
        assert False
    # 2 读取标签
    Ytrain = pd.read_pickle(get_path_feature() + '{}_{}_{}.pkl'.format(train_version, value, 'train'))
    # 3.1 读取训练集合测试集
    Xtrain = pd.read_pickle(get_path_feature() + '{}_{}_{}.pkl'.format(train_version, feature_names[0], 'train'))
    print('the shape of Xtrain is', Xtrain.shape)
    Xtest = pd.read_pickle(get_path_feature() + '{}_{}_{}.pkl'.format(test_version, feature_names[0], 'test'))
    print('the shape of Xtest is', Xtest.shape)
    for feat in feature_names[1:]:
        train_feature = pd.read_pickle(get_path_feature() + '{}_{}_{}.pkl'.format(train_version, feat, 'train'))
        print('the shape of train feature is', train_feature.shape)
        Xtrain = Xtrain.merge(train_feature, on=['ID', 'TimePoint', 'version'], how='left')
        print('the shape of Xtrain is', Xtrain.shape)
        test_feature = pd.read_pickle(get_path_feature() + '{}_{}_{}.pkl'.format(test_version, feat, 'test'))
        print('the shape of test feature is', test_feature.shape)
        Xtest = Xtest.merge(test_feature, on=['ID', 'TimePoint', 'version'], how='left')
        print('the shape of Xtest is', Xtest.shape)
    # 3.2 不要使用有缺失值的样本
    mask1 = Ytrain[value] != -9999
    mask2 = Ytrain[value] != -99999
    mask = mask1 & mask2
    Xtrain = Xtrain[mask].reset_index(drop=True)
    Ytrain = Ytrain[mask].reset_index(drop=True)

    # 4 数据准备完毕， 开始训练预测
    clf = xgb.XGBRegressor(max_depth=3, learning_rate=0.03, n_estimators=200, silent=True, objective='reg:linear')
    clf.fit(Xtrain[feature_names].values, Ytrain[value].values, eval_metric='rmse')
    y_pred = clf.predict(Xtest[feature_names].values)

    # 5 根据是否存在，或者存在但是为NA   
    if os.path.exists(get_path_feature() + get_path_feature() + '{}_{}_{}.pkl'.format(test_version, value, 'test')):
        Ytest = pd.read_pickle(get_path_feature() + '{}_{}_{}.pkl'.format(test_version, value, 'test'))
        print('the shape of Ytest is', Ytest.test)
        for i in range(Ytest.shape[0]):
            if pd.isna(Ytest[value][i]):
                Ytest[value][i] = y_pred[i]
            else:
                pass
        Xtest = Xtest.merge(Ytest, on=['ID', 'TimePoint', 'version'], how='left')
    else:
        Xtest[value] = y_pred

    # 6 预测完毕，保重！
    SaveFeature(feat_df=Xtest, feature_name=value, version=test_version, kind='test')


def run_fill_na():
    feats = ['SBP_value_last_mean', 'DBP_value_last_mean', 'HEART_RATE_TIMES_value_last_mean',
             'GLU_value_last_mean', 'HEIGHT_value_last_mean', 'WEIGHT_value_last_mean', 'BMI_value_last_mean']
    pool = multiprocessing.Pool(7)

    for feat in pool.imap_unordered(fill_na, feats):
        print('finished {}'.format(feat))
