def gen_config_xgboost():
    """
    :return:  model_config
    """

    model_config = {
        "model_name": "xgboost",

        # 2 被包装的模型相关参数
        "model_params":
            {
                'objective': 'binary:logistic',
                'learning_rate': 0.1,
                'eval_metric': 'auc',
                'max_depth': 4,
                'gamma': 0,
                'min_child_weight': 1,
                'lambda': 1,
                'alpha': 0,
                'colsample_bytree': 0.7,
                'subsample': 0.9,
                'colsample_bylevel': 0.7,
                'nthread': 40,
                'silent': True,
                "verbose": 0,
                "seed": 2018,
            },
        # 3 被包装的模型相关训练参数
        "train_params":
            {
                "num_boost_round": 2000,
                "early_stopping_rounds": 30,
            }
    }
    return model_config
