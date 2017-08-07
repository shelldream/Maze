# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")
import numpy as np

def cal_precision_at_K():
    pass

def cal_mrr():
    pass

def cal_err():
    pass


def cal_ndcg(scores, preds, topn=-1):
    """
        计算排序结果topN位置的NDCG
        Args:
            scores: list, 样本实际score
            preds: list, 模型预测score
            topn: int, ndcg的topn位置
        Rets:
            ndcg: float, 排序结果topN 位置的NDCG
    """
    if len(scores) != len(preds):
        raise ValueError("The lengths of scores and preds are not the same!!")
    topn = len(scores) if topn == -1 else min(topn, len(scores))
    scores = scores[0:topn]
     
    score_preds_pairs = [(scores[i], preds[i]) for i in range(len(scores))]
    sorted_pairs_by_score = sorted(score_preds_pairs, key=lambda x:x[0], reverse=True)  #按照原始的score降序排序
    sorted_pairs_by_preds = sorted(score_preds_pairs, key=lambda x:x[1], reverse=True)  #按照预测的score降序排序

    def cal_dcg(sorted_pairs_list):
        sum_dcg = 0.0
        for i in range(len(sorted_pairs_list)):
            score = sorted_pairs_list[i][0]
            discount = 1.0/np.log2(2+i) 
            #sum_dcg += (2**score-1.0)*discount
            sum_dcg += score*discount

        return sum_dcg

    IDCG = cal_dcg(sorted_pairs_by_score)
    DCG = cal_dcg(sorted_pairs_by_preds)
    
    if IDCG == 0.0:
        return 0.0
    
    ndcg = DCG/IDCG
    return ndcg


if __name__ == "__main__":
    scores = [2, 1, 2, 0]
    preds = [4,3,2,1]
    print cal_ndcg(scores, preds) 
