#-*- coding:utf-8 -*-
"""
    Description:  加载数据相关的函数
    Author:  shelldream
    Date: 2017.07.11
"""
import sys
reload(sys).setdefaultencoding('utf-8')
sys.path.append("./utils")
sys.path.append("../utils")
sys.path.append("./metrics")
sys.path.append("../metrics")

import os
import copy
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

from common import *

def load_libsvm_file(filename, isDense=False):
    """
        载入svmlight/libsvm格式的纯文本数据文件
        稀疏表示的数据将被输出为稠密的numpy数组
        Args:
            filename: str, 数据文件列表
            isDense: bool, 数据是否为稠密矩阵表示
        Rets:
            x_data: numpy.array
            y_data: numpy.array
    """
    if not os.path.exists(filename):
        raise ValueError("%s does not exist!"%filename) 
    
    try:
        x_data, y_data = load_svmlight_file(filename)
        if isDense:
            x_data = x_data.todense()
        x_data = np.array(x_data)
        y_data = np.array(y_data)
    except:
        raise ValueError("Fail to load %s in libsvm format"%filename)
    
    return y_data, x_data 
                
def load_csv_with_table_header(file_list, fmap=None, delimiter="\t", black_feature_list=None):
    """
        载入带表头的的文本数据，schema为文件第一行
        Args:
            file_list: list, 数据文件名列表
        Rets:
            filtered_data: pandas dataframe 格式，过滤掉部分特征后的结果
            data: pandas dataframe 格式, 从多个数据文件中读取数据合并后的结果
    """
    for filename in file_list:
        if not os.path.isfile(filename):
            print "Warning! %s does not exist!"%filename
            continue
        data = pd.read_table(filename, sep=delimiter)
    
    frames = []
    for filename in file_list:
        if not os.path.isfile(filename):
            print "Warning! %s does not exist!"%filename
            continue
        df = pd.read_table(filename, sep=delimiter)
        frames.append(df)
    data = pd.concat(frames, axis=0)
    data = data.reset_index(drop=True)  #合并不同数据文件的数据，然后重置index
    filtered_data = filter_feature(data, black_feature_list)

    return filtered_data, data

def load_csv_with_fmap(file_list, fmap_filename="fmap.schema", delimiter="\t", black_feature_list=None):
    """
        载入文本数据,并且根据指定的feature map文件制定数据的schema
        
        Args:
            file_list: list, 数据文件列表,数据格式按照 "\t" 分隔
            fmap_filename: str, feature schema 文件名, 格式：index\tfeature_name\tdata_type
        Rets:
    """
    if not os.path.isfile(fmap_filename):
        raise ValueError("schema file %s does not exist!"%fmap_filename)
    
    dtype_map = {
        "int": np.int32,
        "float": np.float64,
        "str": np.object,
        "string": np.object
    }

    ft_name_type_dict = {}
    ft_name_list = []
    with open(fmap_filename, "r") as f_map_r:
        for line in f_map_r:
            try:
                if "#" not in line: #注释行
                    index, feature_name, data_type = line.rstrip().split("\t")
                    ft_name_list.append(feature_name)
                    ft_name_type_dict[feature_name] = dtype_map[data_type]
                else:
                    continue
            except:
                continue
    #print "feature name list: ", ft_name_list
    #print "feature type dict:", ft_name_type_dict

    frames = []
    for filename in file_list:
        if not os.path.isfile(filename):
            print "Warning! %s does not exist!"%filename
            continue
        df = pd.read_csv(filename, sep=delimiter, header=None, names=ft_name_list, \
            dtype=ft_name_type_dict)
        frames.append(df)
    if len(frames) == 0:
        return None
    data = pd.concat(frames, axis=0)
    data = data.reset_index(drop=True)  #合并不同数据文件的数据，然后重置index
    filtered_data = filter_feature(data, black_feature_list)

    return filtered_data, data

def filter_feature(data, black_feature_list=None):
    """
        Args:
            black_feature_list: list,
            data: pandas dataframe 格式
        Rets:

    """
    if black_feature_list is None:
        return data
    filtered_data = copy.deepcopy(data)
    for feature in black_feature_list:
        try:
            filtered_data.pop(feature)
        except:
            print colors.RED + "%s does not in the feature schema"%feature + colors.ENDC
    return filtered_data

if __name__ == "__main__":
    test_data = load_csv_with_table_header(["test.dat", "test.txt"]) 
    print test_data
    #test_y, test_x = load_libsvm_file(["libsvm.dat"], True)
    #test_y, test_x = load_libsvm_file("./libsvm1.dat")
    #print test_y
    #print test_x 

