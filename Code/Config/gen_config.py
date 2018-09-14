from Code.Model.config_xgboost import gen_config_xgboost
from Code.Model.config_lightgbm import gen_config_lightgbm


def gen_config(versions, config_name, feature_names, model_name, save_log=True):
    """
    :param versions: list, 特征版本号
    :param config_name: str
    :param feature_names: list
    :param model_name: str
    :param save_log: bool
    :return:
    """

    model_config = {}
    if model_name == 'xgboost':
        model_config = gen_config_xgboost()
    elif model_name == 'lightgbm':
        model_config = gen_config_lightgbm()

    config = {
        'config_name': config_name,
        'versions': versions,
        'feature_names': feature_names,
        # 1  验证参数
        'validate_params':
            {'kind': 'kf_by_id',
             'shuffle': True,
             'random_state': 2018,
             'n_splits': 5,
             'norm_feat_imp': True
             },
        # 2 模型配置参数
        'model': model_config,
        # 4 是否测试， 预测测试集合是否oof
        'test': False,
        'test_version': 'v1',
        'oof': True,
        'save_log': save_log}
    return config
