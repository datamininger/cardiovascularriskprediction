from utils import ReadHistData
from utils import compute_stats_missing_dict, SaveFeature, SaveFeatHist, ReadLabelsData, filter_hist_data, ExistFeature
import multiprocessing
from sklearn.feature_extraction import DictVectorizer
import pandas as pd


def gen_missing_feature(stats_name, version='v1', kind='train'):
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
    labels['stats_dict'] = labels.apply(lambda label: compute_stats_missing_dict(filter_hist_data(label, followup), values, stats_name), axis=1)
    v = DictVectorizer()
    stats_matrix = v.fit_transform(labels['stats_dict'].values).toarray()
    value_names = v.get_feature_names()
    feature_names = ['{}_{}'.format(value_name, stats_name) for value_name in value_names]
    stats_df = pd.DataFrame(data=stats_matrix, columns=feature_names)
    labels = pd.concat([labels, stats_df], axis=1)
    #  3 保存特征
    for feat in feature_names:
        SaveFeature(labels, feat, version, kind)
    return feature_names


def run_missing_feature_followup():
    i = 1
    feat_hist_file_name = 'gen_missing_feature'
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
    pool = multiprocessing.Pool(processes=20)
    feat_names = []
    for feat_names_1 in pool.imap_unordered(gen_missing_feature, stats_names):
        feat_names += feat_names_1
    SaveFeatHist(feat_hist_file_name, i, feat_names)
