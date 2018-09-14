import pandas as pd
from utils import ReadHistData
from utils import compute_action_time_feature, SaveFeature, SaveFeatHist, ReadLabelsData, filter_hist_data, ExistFeature
import multiprocessing


def gen_action_time_feature(stats_name, version='v2', kind='train'):
    """
    :param stats_name:
    :param version:
    :param kind:
    :return:
    """

    # 1 读取历史数据
    followup = ReadHistData(info='followup_person_info', version=version, kind=kind)
    labels = ReadLabelsData(version, kind)
    #  2 计算特征
    labels[stats_name] = labels.apply(lambda label: compute_action_time_feature(filter_hist_data(label, followup), stats_name), axis=1)
    SaveFeature(labels, stats_name, version, kind)
    return stats_name


def run_action_time_feature():
    i = 0
    feat_hist_file_name = 'gen_acition_time_feature'
    stats_names = ['time_diff_mean', 'time_diff_max', 'time_diff_min', 'time_diff_range', 'time_diff_std',
                   'time_diff_diff_mean', 'time_diff_diff_max', 'time_diff_diff_min', 'time_diff_diff_range', 'time_diff_diff_std']
    pool = multiprocessing.Pool(processes=10)
    feat_names = []
    for feat_names_1 in pool.imap_unordered(gen_action_time_feature, stats_names):
        feat_names.append(feat_names_1)
    SaveFeatHist(feat_hist_file_name, i, feat_names)

