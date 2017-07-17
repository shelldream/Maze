# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")

import sklearn.metrics as sk_metrics

def cal_accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    """

    """
    return sk_metrics.accuracy_score(y_true, y_pred, normalize, sample_weight)
