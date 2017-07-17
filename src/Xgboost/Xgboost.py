#-*- coding:utf-8 -*-
"""
    Description:
        利用Xgboost进行分类、回归、排序等机器学习任务
    Author: shelldream
    Date: 2017-01-07
"""
import sys
reload(sys).setdefaultencoding('utf-8')
sys.path.append("./utils")
sys.path.append("../utils")
sys.path.append("./metrics")
sys.path.append("../metrics")


import datetime
import xgboost as xgb
import numpy as np
import pickle
from common import *
import metrics.classify_metrics as classify_metrics

class Xgboost(object):
    def __init__(self, params={}, model_saveto="model"):
        """
            Args:
                evals (list of pairs (DMatrix, string)) – List of items to be evaluated during training, \
                    this allows user to watch performance on the validation set. e.g: [(dtrain, "train"), (dtest, "eval")]
                model_saveto: string, 训练完成的model保存目录
            Rets:
                No returns
        """
        self.params = params
        self.model_saveto = model_saveto + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + ".pickle.dat"

class XgbClassifier(Xgboost):
    def __init__(self, params={}, model_saveto="model"):
        super(XgbClassifier, self).__init__()
        self.bst_classifier = xgb.XGBClassifier(**self.params)

    def train(self, x_train, y_train):
        """
            Args:
                x_train:
                y_train:
        """
        model = self.bst_classifier.fit(x_train, y_train) 
        y_pred = model.predict(x_train) 
        accuracy_score = classify_metrics.cal_accuracy_score(y_train, y_pred) 
        print colors.BLUE + "In the training set, classify accuracy score:%f"%accuracy_score + colors.ENDC
        try:
            pickle.dump(model, open(self.model_saveto, "wb"))
            print colors.GREEN + "%s has been saved successfully!!"%self.model_saveto + colors.ENDC
        except:
            print colors.RED + "Warning: %s has not been saved!! Please ensure that the output directory of your model exists. "%self.model_saveto + colors.ENDC

    def predict(self):
        pass
