from utils import ReadHistData, SaveFeature, ReadLabelsData, filter_hist_data
import pandas as pd


def compute_age(df_person, label):
    """
    :param df_person:
    :return:
    """
    age = (label['TimePoint'] - df_person['DATE_OF_BIRTH'].max()).days
    return age


def gen_base_feature(version, kind='train'):
    """
    计算年龄、性别、确认时长
    :param kind:
    :return:
    """
    #  1 读取历史数据
    followup = ReadHistData(info='followup_person_info', version=version, kind=kind)
    labels = ReadLabelsData(version, kind)
    # 1 性别
    labels = labels.merge(followup.groupby('ID')['SEX_CODE'].max().to_frame('SEX_CODE').reset_index(), on=['ID'], how='left')
    SaveFeature(labels, 'SEX_CODE', version, kind)
    # 2 在时间点的年龄
    labels['age'] = labels.apply(lambda label: compute_age(filter_hist_data(label, followup), label), axis=1)
    SaveFeature(labels, 'age', version, kind)
    # 3 确认高血压时的年龄
    labels = labels.merge(
        followup.groupby('ID').apply(lambda df_person:
                                     (df_person['CONFIRM_DATE'].max() - df_person['DATE_OF_BIRTH'].max()).days).to_frame('confirm_age').reset_index(), on=['ID'], how='left')
    SaveFeature(labels, 'confirm_age', version, kind)
    # 4 时间点与确认高血压时间的差
    labels['time_diff_confirm_2TimePoint'] = (labels['age'] - labels['confirm_age'])
    SaveFeature(labels, 'time_diff_confirm_2TimePoint', version, kind)



def gen_base_feature2(version, kind='train'):
    #  1 读取历史数据
    followup = ReadHistData(info='followup_person_info', version=version, kind=kind)
    labels = ReadLabelsData(version, kind)

    # 2 第一次随访的age
    labels = labels.merge(followup.groupby('ID')['DATE_OF_BIRTH'].max().reset_index(), on='ID', how='left')
    first_followup_time_df = followup.groupby('ID')['FOLLOWUP_DATE'].min().to_frame('first_followup_time').reset_index()
    labels = labels.merge(first_followup_time_df, on='ID', how='left')
    labels['first_followup_age'] = (labels['first_followup_time'] - labels['DATE_OF_BIRTH']).dt.days
    SaveFeature(labels, 'first_followup_age', version, kind)

    # 3 第一次随访到确认高血压的时间
    labels = labels.merge(followup.groupby('ID')['CONFIRM_DATE'].max().reset_index(), on='ID', how='left')
    labels['first_followup_time_diff_confirm_time'] = (labels['first_followup_time'] - labels['CONFIRM_DATE']).dt.days
    SaveFeature(labels, 'first_followup_time_diff_confirm_time', version, kind)

    # 4 当前时间到随访的时间
    labels['TimePoint_diff_first_followup_time'] = (labels['TimePoint'] - labels['first_followup_time']).dt.days
    SaveFeature(labels, 'TimePoint_diff_first_followup_time', version, kind)

