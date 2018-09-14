from Code.Model.ModelLightgbm import ModelLightgbm
from Code.Model.ModelXgboost import ModelXgboost


def InitModel(Xtrain, Ytrain, Xtest, model_config):
    """
    :param Xtrain:
    :param Ytrain:
    :param Xtest:
    :param model_config:
    :return:
    """
    assert model_config['model_name'] in ['xgboost', 'lightgbm']
    if model_config['model_name'] == 'xgboost':
        return ModelXgboost(Xtrain, Ytrain, Xtest, model_config)
    elif model_config['model_name'] == 'lightgbm':
        return ModelLightgbm(Xtrain, Ytrain, Xtest, model_config)
