#-*- coding:utf-8 -*-
"""
    Description:
        利用Xgboost进行分类、回归、排序等机器学习任务
        传入的数据格式保证是xgboost.DMatrix格式
    Author: shelldream
    Date: 2017-01-07
"""
import sys
reload(sys).setdefaultencoding('utf-8')
sys.path.append("./utils")
sys.path.append("../utils")

import datetime
import xgboost as xgb
import numpy as np
from common import *

class Xgboost:
    def __init__(self, train_data=None, validation_data=None, test_data=None, params=None, boost_round=100, evals=(), 
            model_saveto="model"):
        """
            Args:
                train_data: DMatrix 格式, 训练数据 
                validation_data: DMatrix 格式, 验证数据 
                test_data: DMatrix 格式, 测试数据
                params: dict, booster 训练参数 
                boost_round: Number of boosting iterations
                evals (list of pairs (DMatrix, string)) – List of items to be evaluated during training, \
                    this allows user to watch performance on the validation set. e.g: [(dtrain, "train"), (dtest, "eval")]
                model_saveto: string, 训练完成的model保存目录
            Rets:
                No returns
        """
        if train_data is not None and type(train_data) != type(xgb.DMatrix([])):
            raise ValueError("Please ensure that the format of the train data is xgboost.core.DMatrix ")
        else:
            self.train_data = train_data
        
        if validation_data is not None and type(validation_data) != type(xgb.DMatrix([])):
            raise ValueError("Please ensure that the format of the validation data is xgboost.core.DMatrix ")
        else:
            self.validation_data = validation_data
         
        if test_data is not None and type(test_data) != type(xgb.DMatrix([])):
            raise ValueError("Please ensure that the format of the test data is xgboost.core.DMatrix ")
        else:
            self.test_data = test_data
        
        self.params = params
        self.boost_round = boost_round
        self.evals = evals
        self.model_saveto = model_saveto + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M")


class XgbClassifier(Xgboost):
    def train(self, is_sklearn_api=False):
        if is_sklearn_api:
            pass
        else:
            self.bst_classifier = xgb.train(params=self.params, dtrain=self.train_data,\
                num_boost_round=self.boost_round, evals=self.evals)
            try:
                self.bst_classifier.save_model(self.model_saveto)
                print colors.GREEN + "%s has been saved successfully!!"%self.model_saveto + colors.ENDC
            except:
                print colors.RED + "Warning: %s has not been saved!! Please ensure that the output directory of your model exists. "%self.model_saveto + colors.ENDC
