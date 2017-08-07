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
import pandas as pd
import pickle
from common import *
import metrics.classification_metrics as classify_metrics
import metrics.regression_metrics as regression_metrics
import metrics.ranking_metrics as ranking_metrics


class Xgboost(object):
    def __init__(self, params=None, model_saveto=None):
        """
            Args:
                params: dict, 初始化模型的参数
                model_saveto: string, 训练完成的model保存目录
            Rets:
                No returns
        """
        self.model = None
        if model_saveto is None:
            self.model_saveto = "./model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + ".txt"
        else:
            self.model_saveto = model_saveto
    
    def cal_feature_importance(self, importance_type="gain"):
        """
            返回按照特征重要性的分析结果
            Args:
                importance_type: str, 特征重要性类型, 目前只支持 "weight"、"gain"、"cover"
            Rets:
                sorted_fscores:
                sorted_scores:
        """
        if self.model is None:
            raise ValueError("The model is empty!!")
        self.fscores = self.model.booster().get_fscore()
        sorted_fscores = sorted(self.fscores.items(), key=lambda x:x[1], reverse=True) 
        self.scores = self.model.booster().get_score(importance_type=importance_type)
        sorted_scores = sorted(self.scores.items(), key=lambda x:x[1], reverse=True)
        
        print colors.BLUE + "-"*30 + " The feature importance of each feature " + "-"*30 + colors.ENDC
        for (feature, value) in sorted_fscores:
            print colors.BLUE + "\t\t\t%s\t%f"%(feature, value) + colors.ENDC
        
        print colors.BLUE + "\n\n" + "-"*30 + " The feature importance score (%s) "%importance_type + "-"*30 + colors.ENDC
        for (feature, value) in sorted_scores:
            print colors.BLUE + "\t\t\t%s\t%f"%(feature, value) + colors.ENDC
        
        return sorted_fscores, sorted_scores
    
    def save_text_model(self, model_saveto=None):
        """Save model in text format"""
        model_saveto = model_saveto if model_saveto is not None else self.model_saveto
        try:
            self.model.booster().dump_model(model_saveto)
            print colors.GREEN + "%s has been saved successfully in raw text format!!"%model_saveto + colors.ENDC
        except:
            print colors.RED + "Warning: %s has not been saved!! Please ensure that the output directory of your model exists. "%model_saveto + colors.ENDC
    
    def save_model(self, model_saveto=None):
        """save model file in binary format"""
        model_saveto = model_saveto if model_saveto is not None else self.model_saveto
        try:
            pickle.dump(self.model, open(model_saveto, "wb"))
            print colors.GREEN + "%s has been saved successfully in binary format!!"%model_saveto + colors.ENDC
        except:
            print colors.RED + "Warning: %s has not been saved!! Please ensure that the output directory of your model exists. "%model_saveto + colors.ENDC
        
    def load_model(self, model_load_from=None):    
        """load model file in binary format"""
        model_load_from = model_load_from if model_load_from is not None else self.model_saveto
        try:
            self.model = pickle.load(open(model_load_from, "rb"))
            print colors.GREEN + "%s has been loaded successfully in binary format!!"%model_load_from + colors.ENDC     
        except:
            raise ValueError(colors.RED+"Model %s faied to be loaded!!"%model_load_from + colors.ENDC)
    

class XgbRanker(Xgboost):
    """
        利用pointwise regression 的方法训练排序模型
    """
    def __init__(self, params=None, model_saveto=None):
        super(XgbRanker, self).__init__()
        self.params = params if params is not None else dict()
        self.model = xgb.XGBRegressor(**self.params)

    def train(self, x_train, y_train, model_saveto=None):
        model_saveto = model_saveto if model_saveto is not None else self.model_saveto
        self.model.fit(x_train, y_train) 
        y_pred = self.model.predict(x_train)
        mean_squared_error = regression_metrics.cal_mean_squared_error(y_train, y_pred)
        print colors.BLUE + "In the training set, mean squared error: %f"%mean_squared_error + colors.ENDC
        self.save_model(model_saveto)
        self.save_text_model(model_saveto + ".raw_text")
        sorted_fscores , sorted_scores = self.cal_feature_importance(importance_type="gain")

    def analysis(self, x_test, raw_test_data, model_load_from=None, groupby=None, target="label"):
        model_load_from = model_load_from if model_load_from is not None else self.model_saveto
        self.load_model(model_load_from)
        y_pred = self.model.predict(x_test)
        raw_test_data.insert(0, "predict_score", y_pred)
        if groupby is not None:
            grouped = raw_test_data.groupby(groupby)
            ndcg_list = []
            ndcg5_list = []
            for key, data in grouped:
                preds = data["predict_score"].tolist() 
                scores = data[target].tolist()
                if len(set(scores)) == 1:  #if the grouped data has only one label, then continue
                    continue
                ndcg = ranking_metrics.cal_ndcg(scores, preds)
                ndcg5 = ranking_metrics.cal_ndcg(scores, preds, 5)
                ndcg_list.append(ndcg)
                ndcg5_list.append(ndcg5)
            average_ndcg = np.mean(ndcg_list) 
            average_ndcg5 = np.mean(ndcg5_list) 
            print colors.BLUE + "In the testing test, group by %s  average_ndcg: %f  average_ndcg@5: %f"%(groupby, average_ndcg, average_ndcg5) + colors.ENDC
        else:
            preds = raw_test_data["predict_score"].tolist() 
            scores = raw_test_data[target].tolist()
            average_ndcg = ranking_metrics.cal_dcg(scores, preds)
            average_ndcg5 = ranking_metrics.cal_dcg(scores, preds, 5)
            print colors.BLUE + "In the testing test, average_ndcg: %f  average_ndcg@5: %f"%(average_ndcg, average_ndcg5) + colors.ENDC
            
    def predict(self, x_data, raw_x_data, predict_result_output, model_load_from=None):
        self.load_model(model_load_from)
        y_pred = self.model.predict(x_data)
        raw_x_data.insert(0, "predict_score", y_pred)
        try:
            raw_x_data.to_csv(predict_result_output, sep="\t", index=False)
            print colors.BLUE + "The predicted ranking score result has been saved as %s"%predict_result_output + colors.ENDC
        except:
            raise ValueError(colors.RED + "Fail to predict the ranking score!! Please check your data and your output path!" + colors.ENDC)
        return y_pred

         
class XgbRegressor(Xgboost):
    def __init__(self, params=None, model_saveto=None, eval_metric=None):
        super(XgbRegressor, self).__init__()
        self.params = params if params is not None else dict()
        self.model = xgb.XGBRegressor(**self.params)
         
    def train(self, x_train, y_train, model_saveto=None):
        self.model = self.model.fit(x_train, y_train) 
        y_pred = self.model.predict(x_train)
        mean_squared_error = regression_metrics.cal_mean_squared_error(y_train, y_pred)
        print colors.BLUE + "In the training set, mean squared error: %f"%mean_squared_error + colors.ENDC
        self.save_model(model_saveto)
        self.save_text_model(model_saveto + ".raw_text")
        
        sorted_fscores , sorted_scores = self.cal_feature_importance(importance_type="gain")
            
    def predict(self, x_data, raw_x_data, predict_result_output, model_load_from=None):
        self.load_model(model_load_from)
        y_pred = self.model.predict(x_data)
        x_data.insert(0, "predict_score", y_pred)
        try:
            x_data.to_csv(predict_result_output, sep="\t", index=False)
            print colors.BLUE + "The regression predict result has been saved as %s"%predict_result_output + colors.ENDC
        except:
            raise ValueError(colors.RED + "Fail to predict the regression score!! Please check your data and your output path!" + colors.ENDC)
        return x_data
         
    def analysis(self, x_test, raw_test_data, model_load_from=None, groupby=None, target="label"):
        self.load_model(model_load_from) 
        y_pred = self.model.predict(x_test)
        mean_squared_error = regression_metrics.cal_mean_squared_error(y_test, y_pred)
        print colors.BLUE + "In the test set, mean squared error: %f"%mean_squared_error + colors.ENDC
        

class XgbClassifier(Xgboost):
    def __init__(self, params=None, model_saveto=None):
        super(XgbClassifier, self).__init__()
        self.params = params if params is not None else dict()
        self.model = xgb.XGBClassifier(**self.params)
    
    def train(self, x_train, y_train, model_saveto=None, importance_type="weight"):
        """
            Args:
                x_train:
                y_train:
        """
        self.model.fit(x_train, y_train) 
        y_pred = self.model.predict(x_train) 
        self.save_model(model_saveto)
        self.save_text_model(model_saveto + ".raw_text")

        accuracy_score = classification_metrics.cal_accuracy_score(y_train, y_pred) 
        print colors.BLUE + "In the training set, classification accuracy score: %f"%accuracy_score + colors.ENDC
        if len(set(y_train)) == 2 and len(set(y_pred)) == 2: #binary classification
            predict_probs = np.array([prob[1] for prob in self.model.predict_proba(x_train)])
            auc = classification_metrics.cal_auc(y_train, predict_probs)
            print colors.BLUE + "In the training set, AUC: %f"%auc + colors.ENDC
        
        sorted_fscores , sorted_scores = self.cal_feature_importance(importance_type="gain")
            
    def predict(self, x_data, raw_x_data, predict_result_output, model_load_from=None):
        self.load_model(model_load_from)
        y_pred = self.model.predict(x_data)
        if len(set(y_train)) == 2 and len(set(y_pred)) == 2: #binary classification
            predict_probs = np.array([prob[1] for prob in self.model.predict_proba(x_train)])
        
        return y_pred
    
    def analysis(self,x_test, raw_test_data, model_load_from=None, groupby=None, target="label"):
        self.load_model(model_load_from) 
        y_pred = self.model.predict(x_test) 

        accuracy_score = classification_metrics.cal_accuracy_score(y_test, y_pred) 
        print colors.BLUE + "In the test set, classification accuracy score: %f"%accuracy_score + colors.ENDC
        if len(set(y_test)) == 2 and len(set(y_pred)) == 2: #binary classification
            predict_probs = np.array([prob[1] for prob in model.predict_proba(x_test)])
            auc = classification_metrics.cal_auc(y_test, predict_probs)
            print colors.BLUE + "In the test set, AUC: %f"%auc + colors.ENDC

