#-*- coding:utf-8 -*-
"""
    Description:
    Author:
    Date:
"""
import sys
reload(sys).setdefaultencoding('utf-8')
import os

import pandas as pd

def load_txt_file(fmap_filename="fmap.schema", file_list):
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


def filter_feature(filter_feature_list=None, data):
	"""
		Args:
			filter_feature_list: list,
			data: pandas dataframe 格式
		Rets:

	"""
