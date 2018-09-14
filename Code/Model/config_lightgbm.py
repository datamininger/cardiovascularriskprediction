def gen_config_lightgbm():
    """

    :return:  model_config
    """
    model_config = {
        "model_name": "lightgbm",
        # 1 被包装的模型相关参数
        "model_params": {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            "learning_rate": 0.1,
            'scale_pos_weight': 1,
            'num_leaves': 26,
            'max_depth': 6,
            'min_child_weight': 0.0,
            'reg_lambda': 0.0,
            'min_split_gain': 0,
            'subsample_freq': 1,
            'subsample': 1,
            'colsample_bytree': 0.9,
            'num_threads': 20,
            "verbose": 1,
        },
        'train_params':
            {'eval_metric': 'auc',
             "early_stopping_rounds": 30}
    }
    return model_config
