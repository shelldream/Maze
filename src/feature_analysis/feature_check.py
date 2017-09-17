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

class FeatureChecker(object):
    """
        目前只支持带schema的数据检查
    """
    def __init__(self, data):
        self.data = data
        
    def check_float_value_roughly(self, feat_name):
        feat_data = self.data[feat_name]
        print "Statistical Key Figures of %s"%feat_name
        total_count = self.data.shape[0]
        not_nan_count = feat_data.count()
        not_nan_ratio = float(not_nan_count)/total_count
        print "not_nan_ratio\t%f" %not_nan_ratio
        res = feat_data.describe()
        print res
        return feat_data
         
    def check_float_value_precisely(self, feat_name, bin_num=7, label_name=None):
        feat_data = self.check_float_value_roughly(feat_name)
        print "histogram"
        raw_feat_data = np.array(feat_data)
        feat_data = zip(raw_feat_data, [0 for i in range(len(feat_data))])
        kmeans = KMeans(n_clusters=bin_num, random_state=0).fit(feat_data)  #1-D kmeans to split the data into specific bucket
        
        bin_min = {k:float("inf") for k in xrange(bin_num)}
        bin_max = {k:float("-inf") for k in xrange(bin_num)}
        bin_sample_cnt = {k:0 for k in xrange(bin_num)}
         
        labels = kmeans.labels_
        
        raw_label_data = None if label_name is None else self.data[label_name]
        value_label_dict = {}

        for i in xrange(len(raw_feat_data)):
            value = raw_feat_data[i]
            label_k = labels[i]
            bin_sample_cnt[label_k] += 1
            if value < bin_min[label_k]:
                bin_min[label_k] = value

            if value > bin_max[label_k]:
                bin_max[label_k] = value
            
            if raw_label_data is not None:
                value_label_dict[value] = value_label_dict.get(value, [])
                value_label_dict[value].append(raw_label_data[i])

        bin_min_max_cnt = []
        for k,v in bin_min.items():
            if bin_sample_cnt[k] > 0:
                bin_min_max_cnt.append((bin_min[k], bin_max[k], bin_sample_cnt[k]))

        bin_min_max_cnt = sorted(bin_min_max_cnt, key=lambda x:x[0], reverse=False)
        
        for item in bin_min_max_cnt: 
            print "%f——%f: %d"%(item[0], item[1], item[2])
        
        x_list = []
        y_list = []
        for k,v_list in value_label_dict.items():
            if len(v_list) > 0:
                x_list.append(k)
                y_list.append(sum(v_list)/len(v_list))
        
        print "Person correlation, p-value"
        corrcoef = stats.pearsonr(x_list, y_list) 
        print corrcoef

if __name__ == "__main__":
    data_dir = "/export/sdb/shelldream/activity_rank/"
    #tmp_file_name = ["20170902.txt"]
    tmp_file_name = ["20170902.txt", "20170903.txt", "20170904.txt", "20170905.txt", "20170906.txt"]
    file_list = [data_dir+filename for filename in tmp_file_name]
    black_feature_list = []

    filtered_data, data = load_data.load_csv_with_fmap(file_list, "fmap.schema")
    
    checker = FeatureChecker(filtered_data);
    """ 
    with open("fmap.schema", "r") as fr:
        for line in fr:
            index, fname, ftype = line.strip().split("\t") 
            if ftype == "float" or ftype == "int":
                checker.check_float_value_precisely(fname, 7, "label")
                print "#"*50
    """
