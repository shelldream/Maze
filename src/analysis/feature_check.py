# -*- coding:utf-8 -*-
"""
    Description: Check the validation of the feature.
    Author: shelldream
    Date: 2017-09-15
"""

import sys
reload(sys).setdefaultencoding("utf-8")
from sklearn.cluster import KMeans
import numpy as np
import scipy.stats as stats

sys.path.append("../utils")
sys.path.append("./utils")
import load_data
sys.path.append("../metrics")
sys.path.append("./metrics")
import classification_metrics
import regression_metrics
import ranking_metrics 

class FeatureChecker(object):
    """
        目前只支持带schema的数据检查
    """
    def __init__(self, data):
        self.data = data
        
    def check_distribution_roughly(self, feat_name):
        feat_data = self.data[feat_name]
        print "Statistical Key Figures of %s"%feat_name
        total_count = self.data.shape[0]
        not_nan_count = feat_data.count()
        not_nan_ratio = float(not_nan_count)/total_count
        print "not_nan_ratio\t%f" %not_nan_ratio
        print  feat_data.describe()
        return feat_data
         
    def check_distribution_precisely(self, feat_name, bin_num=7, label_name=None):
        feat_data = self.check_distribution_roughly(feat_name)
        print "\nThe histogram of %s"%feat_name
        raw_feat_data = np.array(feat_data)
        feat_data = zip(raw_feat_data, [0 for i in range(len(feat_data))])
        kmeans = KMeans(n_clusters=bin_num, random_state=0).fit(feat_data)  #1-D kmeans to split the data into specific bucket
        
        bin_min = {k:float("inf") for k in xrange(bin_num)}
        bin_max = {k:float("-inf") for k in xrange(bin_num)}
        bin_sample_cnt = {k:0 for k in xrange(bin_num)}
         
        labels = kmeans.labels_
        
        for i in xrange(len(raw_feat_data)):
            value = raw_feat_data[i]
            label_k = labels[i]
            bin_sample_cnt[label_k] += 1
            
            if value < bin_min[label_k]:
                bin_min[label_k] = value

            if value > bin_max[label_k]:
                bin_max[label_k] = value
            
        bin_min_max_cnt = []
        for k,v in bin_min.items():
            if bin_sample_cnt[k] > 0:
                bin_min_max_cnt.append((bin_min[k], bin_max[k], bin_sample_cnt[k]))

        bin_min_max_cnt = sorted(bin_min_max_cnt, key=lambda x:x[0], reverse=False)
        
        for item in bin_min_max_cnt: 
            print "%f——%f: %d"%(item[0], item[1], item[2])
        
        
    def cal_correlation(self, feat_name, label_name, groupby=None, metrics_func=None):
        """
            Calculate the correlation between the feature and label.
            The data type of the label should be float.
        """
        labels = self.data[label_name].tolist()
        feats = self.data[feat_name].tolist()
        
        feat_label = zip(feats, labels)
         
        value_label_dict = {}
        
        for (value, label) in feat_label:
            value_label_dict[value] = value_label_dict.get(value, [])
            value_label_dict[value].append(float(label))
        
        x_list = []
        y_list = []
        for k,v_list in value_label_dict.items():
            if len(v_list) > 0:
                x_list.append(k)
                y_list.append(sum(v_list)/len(v_list))
        
        print "(Person correlation, p-value) of %s"%feat_name
        corrcoef = stats.pearsonr(x_list, y_list) 
        print corrcoef
        
        if metrics_func is not None:
            print "The metrics of %s"%feat_name
            if groupby is None:
                metrics = metrics_func(labels, feats)
            else:
                grouped = self.data.groupby(groupby)
                metrics_list = []
                for key, data in grouped:
                    values = data[feat_name].tolist()
                    scores = data[label_name].tolist()
                    if len(set(scores)) == 1:
                        continue
                    metrics_list.append(metrics_func(scores, values))
                
                if len(metrics_list) > 0:
                    metrics = sum(metrics_list)/len(metrics_list)
            print metrics
    
if __name__ == "__main__":
    data_dir = "/export/sdb/shelldream/coupon_rank/"
    #tmp_file_name = ["20170904.txt"]
    tmp_file_name = ["20170901.txt", "20170902.txt"]
    #tmp_file_name = ["20170902.txt", "20170903.txt", "20170904.txt", "20170905.txt", "20170901.txt"]
    file_list = [data_dir+filename for filename in tmp_file_name]
    black_feature_list = []

    filtered_data, data = load_data.load_csv_with_fmap(file_list, "fmap.schema")
     
    checker = FeatureChecker(filtered_data);
    with open("fmap.schema", "r") as fr:
        for line in fr:
            index, fname, ftype = line.strip().split("\t") 
            if ftype == "float" or ftype == "int":
                checker.check_distribution_precisely(fname, label_name="label")
                checker.cal_correlation(fname, label_name="label", groupby="pvid", metrics_func=classification_metrics.cal_auc)
                print "#"*50
