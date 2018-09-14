import numpy as np
from utils import ReadHistData, ReadLabelsData, filter_hist_data
import pandas as pd
from utils import get_path_labels


def compute_hist_data_type(df_person):
    """
    根据病人的历史数据判断是哪种样本
    第一种： 不在followup中出现过的
    第二种：出现过但是七个指标全为NA
    第三种：出现一次的且七个指标不全为NA
    第四种：以上三种的合并之后的补集
    :param df_person:
    :return:
    """
    df_person = df_person.reset_index(drop=True)
    if df_person.shape[0] == 0:
        return '1'
    elif df_person.shape[0] > 0:
        mask_vector = df_person[['SBP', 'DBP', 'HEART_RATE_TIMES', 'GLU', 'HEIGHT', 'WEIGHT', 'BMI']].notnull().sum(axis=0).values == np.array([0, 0, 0, 0, 0, 0, 0])
        if np.sum(mask_vector) == 7:
            return '2'
        elif df_person.shape[0] == 1:
            return '3'
        else:
            return '4'
    else:
        assert False


def run_hist_data_type(version, kind):
    """
    :param version:
    :param kind:
    :return:
    """
    hist_data = ReadHistData(version=version, info='followup_person_info', kind=kind)
    labels = ReadLabelsData(version=version, kind=kind)
    labels['data_type'] = labels.apply(lambda label: compute_hist_data_type(filter_hist_data(label=label, followup=hist_data)), axis=1)

    for date_type in ['1', '2', '3', '4']:
        mask = (hist_data['date_type'] == date_type)
        pd.Series(mask).to_pickle(get_path_labels() + '{}_mask_{}_{}.pkl'.format(version, date_type, kind))
    return
