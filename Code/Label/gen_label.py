import pandas as pd
from utils import ReadHistData, SaveLabelsData


def gen_label_person(df_person, min_range_day_CHD=0, max_range_day_CHD=36500, min_range_day_Non_CHD=120, max_range_day_Non_CHD=36500, step_CHD=120, step_Non_CHD=120, version='v1'):
    """
    使用固定步长打标
    首先判断时间范围是否足够，够的话那么从第一个最小的时间点开始打标，逻辑终于清晰了
    :param df_person: pd.Datsframe
    :param min_range_day_CHD: int, 对于CHD来说，约束病人业务时间范围， 单位天, min_range_day_CHD=0意味着不约束时间范围，那么单样本的人也可以打标
    :param max_range_day_CHD: int, 对于CHD来说，约束病人业务时间范围， 单位天,
    :param min_range_day_Non_CHD: int, 对于非CHD来说，最小的时间范围，单位天
    :param max_range_day_Non_CHD: int, 对于非CHD来说，最小的时间范围，单位天
    :param step_CHD: int 对于CHD来说，时间步长， 单位天, 约束病人业务时间范围， 单位天, min_range_day_Non_CHD=0意味着不约束时间范围，那么单样本的人也可以打标
    :param step_Non_CHD: int, 对于非CHD来说，时间步长，单位天
    :param version: int or str, 版本号
    :return:[[]], [ID, TimePoint, version, label]
    """

    # 0 罐子
    label_list = []
    df_person = df_person.reset_index(drop=True)

    # 1 时间处理
    min_range_day_CHD = pd.Timedelta(value=min_range_day_CHD, unit='D')
    max_range_day_CHD = pd.Timedelta(value=max_range_day_CHD, unit='D')
    min_range_day_Non_CHD = pd.Timedelta(value=min_range_day_Non_CHD, unit='D')
    max_range_day_Non_CHD = pd.Timedelta(value=max_range_day_Non_CHD, unit='D')
    step_CHD = pd.Timedelta(value=step_CHD, unit='D')
    step_Non_CHD = pd.Timedelta(value=step_Non_CHD, unit='D')
    three_years = pd.Timedelta(value=3 * 365, unit='D')

    # 2 变量准备
    ID = df_person['ID'].max()
    CHD_FLAG = df_person['CHD_FLAG'].max()
    CHD_DATE = df_person['CHD_DATE'].max()
    min_time = df_person['FOLLOWUP_DATE'].min()
    max_time = df_person['FOLLOWUP_DATE'].max()
    range_day = (max_time - min_time).days
    print('The range of days is (  {}  )'.format(range_day))

    assert CHD_FLAG in [0, 1]
    # 3 开始打标
    if CHD_FLAG == 1:
        """
        min_time ------> TimePoint-------------> max_time
                                 <----------------------------------> 
                                               3 years
                 left: CHD_DATE              middle: CHD_DATE               right: CHD_DATE 
        """
        # 首先看样本时间范围是否达到的要求
        if (range_day >= min_range_day_CHD.days) & (range_day <= max_range_day_CHD.days):
            TimePoint = min_time + min_range_day_CHD
            # 只有该时间点在业务时间内才是有效的，循环是为了判断有没有资格打标
            # 加setp_CHD是为了包括所有样本；
            while TimePoint < (max_time + step_CHD):
                # 根据CHD_DATE在三个时间区间中哪一个来判断是否打标，打什么标
                if CHD_DATE < TimePoint:
                    # 最左边的时间区间，pass
                    pass
                elif (CHD_DATE > TimePoint) & (CHD_DATE < (TimePoint + three_years)):
                    # 中间的时间区间，打标1，进罐
                    label = [ID, TimePoint, version, 1]
                    label_list.append(label)
                elif CHD_DATE > (TimePoint + three_years):
                    # 最右边的时间区间，打标0，进罐
                    label = [ID, TimePoint, version, 0]
                    label_list.append(label)
                # 向前移动step_CHD，继续打标
                TimePoint += step_CHD
        else:
            pass

    else:
        """
        min_time ------> TimePoint-------------> max_time
                                 <----------------------------------> 
                                               3 years        
        """
        if (range_day >= min_range_day_Non_CHD.days) & (range_day <= max_range_day_Non_CHD.days):
            TimePoint = min_time + min_range_day_Non_CHD
            # 直接看3年时间段是否在业务时间内，循环是为了判断有没有资格打标
            while (TimePoint + three_years) <= max_time:
                # 打标0，进罐
                label = [ID, TimePoint, version, 0]
                label_list.append(label)
                # 向前移动 step_Non_CHD， 继续打标
                TimePoint += step_Non_CHD
        else:
            pass
    return label_list


def gen_label_by_followup(min_range_day_CHD=0, max_range_day_CHD=36500, min_range_day_Non_CHD=120, max_range_day_Non_CHD=36500, step_CHD=120, step_Non_CHD=120, version='v1'):
    followup = ReadHistData(info='followup', kind='train')
    df = followup.groupby('ID').apply(lambda df_person: gen_label_person(df_person=df_person,
                                                                         min_range_day_CHD=min_range_day_CHD,
                                                                         max_range_day_CHD=max_range_day_CHD,
                                                                         min_range_day_Non_CHD=min_range_day_Non_CHD,
                                                                         max_range_day_Non_CHD=max_range_day_Non_CHD,
                                                                         step_CHD=step_CHD,
                                                                         step_Non_CHD=step_Non_CHD,
                                                                         version=version)).to_frame('labels').reset_index()
    labels = list(df['labels'].values)
    label_list = []
    for label in labels:
        if len(label) > 0:
            label_list += label
    df_labels = pd.DataFrame(label_list, columns=['ID', 'TimePoint', 'version', 'label'])
    params_dict = {
        'min_range_day_CHD': min_range_day_CHD,
        'max_range_day_CHD': max_range_day_CHD,
        'min_range_day_Non_CHD': min_range_day_Non_CHD,
        'max_range_day_Non_CHD': max_range_day_Non_CHD,
        'step_CHD': step_CHD,
        'step_Non_CHD': step_Non_CHD,
        'version': version}
    SaveLabelsData(df_labels, params_dict, kind='train')
    return df_labels


"""
def test_gen_label():
    # 边界测试
    # test 1
    df_person = pd.DataFrame({'ID': 10086, 'FOLLOWUP_DATE': pd.Timestamp('2011/10/20'), 'CHD_FLAG': 1, 'CHD_DATE': pd.Timestamp('2012/10/20')}, index=[0])
    print(gen_label(df_person))
    # [[10086, Timestamp('2012-02-17 00:00:00'), 1, 1]]

    # test 2
    df_person = pd.DataFrame({'ID': 10086, 'FOLLOWUP_DATE': pd.Timestamp('2011/10/20'), 'CHD_FLAG': 1, 'CHD_DATE': pd.Timestamp('2011/10/20')}, index=[0])
    print(gen_label(df_person))
    # []

test_gen_label()
"""
