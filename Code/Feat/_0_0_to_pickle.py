import pandas as pd
from utils import  get_path_raw_data


def followup_to_pickle(kind='train'):
    """
    :param kind:
    :return:
    """
    # 读取数据
    followup = pd.read_csv(get_path_raw_data() + '{}_followup.csv'.format(kind))
    followup['FOLLOWUP_DATE'] = pd.to_datetime(followup['FOLLOWUP_DATE'])
    # 按时间排序
    followup = followup.sort_values(by=['FOLLOWUP_DATE'])
    # 保存结果
    followup.to_pickle(get_path_raw_data() + '{}_followup.pkl'.format(kind))


def personinfo_to_pickle(kind='train'):
    """
    :param kind:
    :return:
    """
    # 读取数据
    personinfo = pd.read_csv(get_path_raw_data() + '{}_personinfo.csv'.format(kind))
    # 时间处理
    personinfo['DATE_OF_BIRTH'] = pd.to_datetime(personinfo['DATE_OF_BIRTH'])
    personinfo['CONFIRM_DATE'] = pd.to_datetime(personinfo['CONFIRM_DATE'])
    personinfo['CHD_DATE'] = pd.to_datetime(personinfo['CHD_DATE'])
    # 保存数据
    personinfo.to_pickle(get_path_raw_data() + '{}_personinfo.pkl')
