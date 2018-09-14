from utils import ReadHistData
from utils import SaveFeature, ReadLabelsData, filter_hist_data
import multiprocessing



def compute_missing_ratio(df_person, value):
    df_person = df_person.groupby('FOLLOWUP_DATE')[value].mean()
    num_na = df_person[value].isnull().sum()
    count = df_person.shape[0] + 0.00000001
    missing_ratio = num_na / count
    return missing_ratio


def gen_missing_ratio(value, version='v1', kind='train'):
    """
    :param value:
    :param version:
    :param kind:
    :return:
    """
    # 0 特征名
    feature_name = '{}_missing_ratio'.format(value)
    # 1 读取历史数据
    followup = ReadHistData(info='followup_person_info', version=version, kind=kind)
    labels = ReadLabelsData(version, kind)
    #  2 计算特征
    labels[feature_name] = labels.apply(lambda label: compute_missing_ratio(filter_hist_data(label, followup), value), axis=1)
    # 3 保存特征
    SaveFeature(labels, feature_name, version, kind)
    return


def run_missing_feature():
    pool = multiprocessing.Pool(processes=7)
    values = ['SBP', 'DBP', 'HEART_RATE_TIMES', 'GLU', 'HEIGHT', 'WEIGHT', 'BMI']
    for feat_names in pool.imap_unordered(gen_missing_ratio, values):
        print(feat_names)
