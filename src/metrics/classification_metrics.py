# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")

import sklearn.metrics as sk_metrics

def cal_accuracy_score(labels, y_pred, normalize=True, sample_weight=None):
    """

    """
    return sk_metrics.accuracy_score(labels, y_pred, normalize, sample_weight)


def cal_auc_pr(labels, preds, pos_label=1):
    """Precision Recall curve auc"""
    pass


def cal_auc(labels, preds, pos_label=1):
    """
        ROC AUC
    """
    fpr, tpr, thresholds = sk_metrics.roc_curve(labels, preds, pos_label=pos_label)
    auc = sk_metrics.auc(fpr, tpr)
    return auc


def cal_auc_v2(labels, preds, pos_label=1):
    label_score_pairs = [(labels[i], preds[i]) for i in range(len(labels))]
    label_score_pairs =  sorted(label_score_pairs, key=lambda x:x[1], reverse=True) #按照score降序排序
    
    p_num = 0
    n_num = 0
    for pair in label_score_pairs:
        if pair[0] == 1:
            p_num += 1
        else:
            n_num += 1
    
    """
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
    """
    tpr0 = 0
    fpr0 = 0
    tp = 0
    fp = 0
    auc = 0
    for pair in label_score_pairs:
        if pair[0] == 1:
            tp += 1.0
            tpr1 = tp/p_num
            fpr1 = fp/n_num       
            auc += (tpr0+tpr1)*(fpr1-fpr0)/2
            tpr0 = tpr1
            fpr0 = fpr1
        else:
            fp += 1.0

    auc += 1.0-fpr0
    return auc    

