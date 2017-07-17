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
    def __init__(self, params={}, model_saveto=None):
        """
            Args:
                evals (list of pairs (DMatrix, string)) – List of items to be evaluated during training, \
                    this allows user to watch performance on the validation set. e.g: [(dtrain, "train"), (dtest, "eval")]
                model_saveto: string, 训练完成的model保存目录
            Rets:
                No returns
        """
        self.params = params
        if model_saveto is None:
            self.model_saveto = "./model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + ".pickle.dat"
        else:
            self.model_saveto = model_saveto

class XgbRegressor()
    def __init__(self, params={}, model_saveto=None):
        super(XgbRegressor, self).__init__()
        self.bst_classifier = xgb.XGBRegressor()(**self.params)
        self.model = None
        

class XgbClassifier(Xgboost):
    def __init__(self, params={}, model_saveto=None):
        super(XgbClassifier, self).__init__()
        self.bst_classifier = xgb.XGBClassifier(**self.params)
        self.model = None
        
    def train(self, x_train, y_train, model_saveto=None):
        """
            Args:
                x_train:
                y_train:
        """
        self.model_saveto = model_saveto if model_saveto is not None else self.model_saveto
        self.model = self.bst_classifier.fit(x_train, y_train) 
        y_pred = self.model.predict(x_train) 

        accuracy_score = classify_metrics.cal_accuracy_score(y_train, y_pred) 
        print colors.BLUE + "In the training set, classify accuracy score: %f"%accuracy_score + colors.ENDC
        if len(set(y_train)) == 2 and len(set(y_pred)) == 2: #binary classification
            predict_probs = np.array([prob[1] for prob in self.model.predict_proba(x_train)])
            auc = classify_metrics.cal_auc(y_train, predict_probs)
            print colors.BLUE + "In the training set, AUC: %f"%auc + colors.ENDC

        try:
            pickle.dump(self.model, open(self.model_saveto, "wb"))
            print colors.GREEN + "%s has been saved successfully!!"%self.model_saveto + colors.ENDC
        except:
            print colors.RED + "Warning: %s has not been saved!! Please ensure that the output directory of your model exists. "%self.model_saveto + colors.ENDC

    def predict(self, x_data, model_load_from=None):
        """
        """
        if model_load_from is None:
            model = self.model
        else:
            model = pickle.load(open(model_load_from, "rb"))
        return model.predict(x_data)

    
    def analysis(self,x_test, y_test, model_load_from=None):
        model_load_from = model_load_from if model_load_from is not None else self.model_saveto
        try:
            model = pickle.load(open(model_load_from, "rb"))
            print colors.GREEN + "%s has been loaded successfully!!"%model_load_from + colors.ENDC
        except:
            raise ValueError(colors.RED+"Model %s faied to be loaded!!"%model_load_from + colors.ENDC)
        
        y_pred = model.predict(x_test) 

        accuracy_score = classify_metrics.cal_accuracy_score(y_test, y_pred) 
        print colors.BLUE + "In the test set, classify accuracy score: %f"%accuracy_score + colors.ENDC
        if len(set(y_test)) == 2 and len(set(y_pred)) == 2: #binary classification
            predict_probs = np.array([prob[1] for prob in model.predict_proba(x_test)])
            auc = classify_metrics.cal_auc(y_test, predict_probs)
            print colors.BLUE + "In the test set, AUC: %f"%auc + colors.ENDC

