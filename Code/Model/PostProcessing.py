# coding utf-8

import json
from utils import get_path_log, get_path_submission, get_path_model_file
import numpy as np
import pdb
from utils import compute_validate_result_lightgbm, compute_validate_result_xgboost


class PostProcessing:
    """
    用于后处理模型线下验证，线上预测结果的类
    """

    def __init__(self, booster_offline_list, eval_result_list, submission_online):
        self.booster_offline_list = booster_offline_list
        self.eval_result_list = eval_result_list
        self.validate_results = []
        self.submission_online = submission_online
        self.mean_score_offline = 0.0

    def gen_validate_results(self, model_name, norm_feat_imp, feature_names):
        """
        # 整理验证结果
        :return: list validate_results
        validate_results = [validate_result1, ...]
        validate_result is a dict, like {'best_iteration': xx
                       'score_offline': xx,
                       'feature_importance_dict': xx}
        """
        if model_name == 'lightgbm':
            for booster, eval_result in zip(self.booster_offline_list, self.eval_result_list):
                validate_result = compute_validate_result_lightgbm(booster, eval_result, norm_feat_imp, feature_names)
                self.validate_results.append(validate_result)

        elif model_name == 'xgboost':
            for booster, eval_result in zip(self.booster_offline_list, self.eval_result_list):
                validate_result = compute_validate_result_xgboost(booster, eval_result, norm_feat_imp)
                self.validate_results.append(validate_result)
        return self.validate_results

    def save_validate_model(self, config_name, model_name):

        # 4 保存线下模型
        for i, booster in enumerate(self.booster_offline_list):
            booster.save_model(get_path_model_file() + '{}_{}_{}.m'.format(config_name, model_name, i))

    def save_log(self, config):
        #  1 生成训练日志并保存
        log = {'result': {
            'num_feature': len(config['feature_names']),
            'best_iteration': [validate_result['best_iteration'] for validate_result in self.validate_results],
            'score_offline': [validate_result['score_offline'] for validate_result in self.validate_results],
            'score_mean_offline': np.mean([validate_result['score_offline'] for validate_result in self.validate_results]),
            'score_std_offline': np.std([validate_result['score_offline'] for validate_result in self.validate_results]),
            'fold_results': self.validate_results},
            'config': config
        }
        json.dump(log, open(get_path_log() + '{}.json'.format(config['config_name']), 'w'), indent=2)
        # 2  打印分数
        print('score is : {}'.format(log['result']['score_offline']))
        print('mean score is : {}'.format(log['result']['score_mean_offline']))

    def save_submission(self, config_name):
        # 3 保存提交文件
        self.submission_online.to_csv(get_path_submission() + 'Submission_{}.csv'.format(config_name), index=False)
        return
