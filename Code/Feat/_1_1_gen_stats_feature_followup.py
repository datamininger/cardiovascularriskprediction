from utils import ReadHistData
from utils import compute_stats_value_dict, SaveFeature, SaveFeatHist, ReadLabelsData, filter_hist_data, ExistFeature
import multiprocessing
from sklearn.feature_extraction import DictVectorizer
import pandas as pd


def gen_stats_followup(stats_name, version='v1', kind='train', agg='mean'):
    """
    :param stats_name:
    :param version:
    :param kind:
    :return:
    """
    # 0 被统计的值
    values = ['SBP', 'DBP', 'HEART_RATE_TIMES', 'GLU', 'HEIGHT', 'WEIGHT', 'BMI']
    # 1 读取历史数据
    followup = ReadHistData(info='followup_person_info', version=version, kind=kind)
    labels = ReadLabelsData(version, kind)
    #  2 计算特征
    labels['stats_dict'] = labels.apply(lambda label: compute_stats_value_dict(filter_hist_data(label, followup), values, stats_name, agg), axis=1)
    v = DictVectorizer()
    stats_matrix = v.fit_transform(labels['stats_dict'].values).toarray()
    value_names = v.get_feature_names()
    feature_names = ['{}_{}_{}'.format(value_name, stats_name, agg) for value_name in value_names]
    stats_df = pd.DataFrame(data=stats_matrix, columns=feature_names)
    labels = pd.concat([labels, stats_df], axis=1)
    #  3 保存特征
    for feat in feature_names:
        SaveFeature(labels, feat, version, kind)


def run_stats_followup():
    i = 0
    feat_hist_file_name = 'gen_stats_feature_followup'
    feats = ['SBP', 'DBP', 'HEART_RATE_TIMES', 'GLU', 'HEIGHT', 'WEIGHT', 'BMI']
    stats_names = ['mean', 'max', 'min', 'range', 'std']
    param_list = []
    for stats_name in stats_names:
        for feat in feats:
            param_list.append([stats_name, feat])
    pool = multiprocessing.Pool(processes=20)
    feat_names = []
    for feat_names_1 in pool.imap_unordered(gen_stats_followup, param_list):
        feat_names.append(feat_names_1)
    SaveFeatHist(feat_hist_file_name, i, feat_names)
