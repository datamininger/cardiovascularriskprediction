from utils import SaveFeature, ReadLabelsData, ReadHistData

def gen_time_base_feature(version, kind):
    """
    :param version:
    :param kind:
    :return:
    """
    labels = ReadLabelsData(version=version, kind=kind)
    hist_data = ReadHistData(version=version, info='followup_person_info', kind=kind)


