#-*- coding:utf-8 -*-
"""
    Description: 
    Author: shelldream
    Date: 2017-01-07
"""
import sys
reload(sys).setdefaultencoding('utf-8')

import datatime
import xgboost as xgb

class Xgboost:
    def __init__(self, train_data, test_data, params, boost_round=100, evals=[], 
            model_saveto="model"):
        """
            Args:
                train_data: DMatrix 格式, 训练数据 
                test_data: DMatrix 格式, 测试数据
                params: dict, booster 训练参数 
                boost_round: Number of boosting iterations
                evals (list of pairs (DMatrix, string)) – List of items to be evaluated during training, \
                    this allows user to watch performance on the validation set. e.g: [(dtrain, "train"), (dtest, "eval")]
                model_saveto: 训练完成的model保存目录
            Rets:
                No returns
        """
        self.train_data = train_data
        self.test_data = test_data
        self.param = param
        self.boost_round = boost_round
        self.evals = evals
        self.model_saveto = model_saveto + "/" + datatime.datatime.now().strftime("%Y%m%d-%H%M")


class XgboostClassifier(Xgboost):
    def train(self, is_sklearn_api=False):
        if is_sklearn_api:
            pass
        else:
            self.bst_classifier = xgb.train(self.param, self.train_data, 
                num_boost_round=self.boost_round, evals=self.evals)
            print self.bst_classifier.eval(self.test_data)
            self.bst_classifier.save_model(self.model_saveto)

