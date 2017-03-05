#-*- coding:utf-8 -*-
"""
    Description:
    Author:
    Date:
"""
import sys
reload(sys).setdefaultencoding('utf-8')
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file


def load_libsvm_txt_file(isDense=True, file_list=None):
    """
        载入libsvm格式的纯文本数据文件
        Args:
            file_list: list, 数据文件列表
            isDense: bool, 数据是否为稠密矩阵表示
        Rets:
            x_data: numpy.array
            y_data: numpy.array
    """
    if file_list is None:
        raise ValueError("file_list is empty")
    
    x_data = None
    y_data = None
    for filename in file_list:
        try:
           x_tmp, y_tmp = load_svmlight_file(filename)
        except:
            print "Fail to load %s in libsvm format"%filename
        x_tmp = np.array(x_tmp)
        y_tmp = np.array(y_tmp)

        if not isDense:
            x_tmp.todense()
        if x_data is None:
            x_data = x_tmp
            y_data = y_tmp
        else:
            x_data = np.row_stack((x_data, x_tmp))
            y_data = np.row_stack((y_data, y_tmp))
    
    return y_data, x_data 
                


def load_txt_file(file_list, fmap_filename="fmap.schema"):
    """
        载入文本数据
        
        Args:
            file_list: list, 数据文件列表,数据格式按照 "\t" 分隔
            fmap_filename: str, feature schema 文件, 格式：index\tfeature_name\tdata_type
        Rets:
            data: pandas dataframe 格式, 从多个数据文件中读取数据合并后的结果
    """

    if not os.path.isfile(fmap_filename):
        raise ValueError("%s does not exist!"%fmap_filename)

    ft_name_type_dict = {}
    ft_name_list = []
    with open(fmap_filename, "r") as f_map_r:
        for line in f_map_r:
            try:
                index, feature_name, data_type = line.rstrip("\t")
                ft_name_list.append(feature_name)
                ft_name_type_dict[feature_name] = data_type
            except:
                pass

    frames = []
    for filename in file_list:
        if not os.path.isfile(filename):
            raise ValueError("%s does not exist!"%filename)
        df = pd.read_csv(filename, sep="\t", header=None, names=ft_name_list, \
            dtype=ft_name_type_dict)
        frames.append(df)
    data = pd.concat(frames, axis=0)
    data = data.reset_index(drop=True)  #合并不同数据文件的数据，然后重置index

    return data


def filter_feature(data, filter_feature_list=None):
    """
        Args:
            filter_feature_list: list,
            data: pandas dataframe 格式
        Rets:

    """
