# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")

sys.path.append("../metrics")
sys.path.append("./metrics")
import classification_metrics
import regression_metrics
import ranking_metrics 
import analysis.FeatureChecker
import utils.load_data

data_dir = "/export/sdb/shelldream/coupon_rank/"
filename_list = ["20170901.txt", "20170902.txt", "20170903.txt"]
black_feature_list = []

file_list = [data_dir+filename for filename in filename_list]

filtered_data, data = utils.load_data.load_csv_with_fmap(file_list, "fmap.schema")

checker = analysis.FeatureChecker.FeatureChecker(filtered_data);
with open("fmap.schema", "r") as fr:
    for line in fr:
        index, fname, ftype = line.strip().split("\t") 
        if ftype == "float" or ftype == "int":
            checker.check_distribution_precisely(fname, label_name="label")
            checker.cal_correlation(fname, label_name="label", groupby="pvid", metrics_func=classification_metrics.cal_auc)
            print "#"*50
