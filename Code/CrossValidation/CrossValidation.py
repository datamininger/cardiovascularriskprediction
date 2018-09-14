from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np


def cross_validation_by_ID(Xtrain, validate_params):
    """
    :param Xtrain: pd.DataFrame
    :param validate_params: dict
    :return: 
    """
    train_test_index = []
    kf = KFold(n_splits=validate_params['n_splits'], shuffle=validate_params['shuffle'], random_state=validate_params['random_state'])
    IDs = Xtrain['ID'].unique()
    num_ID = len(IDs)
    kf.get_n_splits(np.zeros(num_ID), IDs)

    for train_ID_index, test_ID_index in kf.split(np.zeros(num_ID), IDs):
        train_ID = IDs[train_ID_index]
        test_ID = IDs[test_ID_index]
        train = Xtrain[Xtrain['ID'].isin(train_ID)]
        test = Xtrain[Xtrain['ID'].isin(test_ID)]
        train_index = list(train.index)
        test_index = list(test.index)
        train_test_index.append([train_index, test_index])
    return train_test_index


def gen_train_validate_index(Xtrain, Ytrain, validate_params):
    """
    支持三种方式交叉验证，1 样本的SKF  2 KF  3 人头的KF
    :param Xtrain: pd.DataFrame
    :param Ytrain: pd.DataFrame
    :param validate_params: dict
    :return: train_test_index
    """
    assert validate_params['kind'] in ['skf', 'kf', 'kf_by_id']
    if validate_params['kind'] == 'skf':
        skf = StratifiedKFold(n_splits=validate_params['n_splits'], shuffle=validate_params['shuffle'], random_state=validate_params['random_state'])
        n_samples = Ytrain.shape[0]
        skf.get_n_splits(np.zeros(n_samples), Ytrain['label'].values)
        train_test_index = skf.split(np.zeros(n_samples), Ytrain['label'].values)
        return train_test_index
    elif validate_params['kind'] == 'kf':
        kf = KFold(n_splits=validate_params['n_splits'], shuffle=validate_params['shuffle'], random_state=validate_params['random_state'])
        n_samples = Ytrain.shape[0]
        kf.get_n_splits(np.zeros(n_samples), Ytrain['label'].values)
        train_test_index = kf.split(np.zeros(n_samples), Ytrain['label'].values)
        return train_test_index
    elif validate_params['kind'] == 'kf_by_id':
        train_test_index = cross_validation_by_ID(Xtrain, validate_params)
        return train_test_index




