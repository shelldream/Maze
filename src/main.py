#-*- coding:utf-8 -*-
"""
    Description: 
        main.py
    Author: shelldream
    Date: 2017-01-07
"""
import sys
reload(sys).setdefaultencoding('utf-8')
import argparse

import xgboost as xgb

import Xgboost.Xgboost
import utils.load_data as ld
from utils.common import *

def main():
    parser = argparse.ArgumentParser(description="This is Maze platform!")

    #model type
    parser.add_argument("--model", choices=["xgboost", ], dest="model", default="xgboost", help="Choose the model you want to use.\
        The choices include xgboost.")

    #task type
    parser.add_argument("--task", choices=["regression", "classification", "ranking"], dest="task", default="classification",
        help="Choose the task type you want to use! The choices include regression, classification, ranking")

    #task mode
    parser.add_argument("--mode", choices=["train", "predict", "analysis"], dest="mode", default="train", \
        help="Choost the task mode you want to use! The choices include train, predict, analysis")

    #data type
    parser.add_argument("--data_type", choices=["libsvm", "csv_with_schema", "csv_with_table_header"], dest="data_type", default="libsvm", help="Choose the data type you want to use.\
        The choices include libsvm, csv_with_schema, csv_without_schema")
    parser.add_argument("--isDense", choices=[True, False], dest="isDense", default=True, help="Set the format of libsvm is dense or not.")
    
    #data path
    parser.add_argument("--train_data_path", action="append", dest="train_data", default=None, help="Choose the \
        training data path. You can choose more than one file path.")
    parser.add_argument("--test_data_path", action="append", dest="test_data", default=None, help="Choose the \
        test data path. You can choose more than one file path.")
    parser.add_argument("--validation_data_path", action="append", dest="validation_data", default=None, help="Choose the \
        validation data path. You can choose more than one file path.")
    parser.add_argument("--predict_data_path", action="append", dest="predict_data", default=None, help="Choose the \
        predict data path. You can choose more than one file path.")
    parser.add_argument("--schema_file", default="fmap.schema", dest="fmap", help="Choose your feature schema file.You only need one file if you choose csv_with_schema data type.")


    #parameter
    parser.add_argument("--black_feature_list", action="append", dest="black_feature_list", default=None, help="Choose the \
    black feature you don't need!")

    parser.add_argument("--parameters", default="{}", dest="parameters", help="Choose the parameters for your model and \
        the format of your parameters is dict format!")
    parser.add_argument("--boost_round", default=100, dest="boost_round", help="Number of boosting iterations")
    
    #model path
    parser.add_argument("--model_path", default="./model/default_model", dest="model_path", help="Set your model path for model saving or loading")
    
    parse_args(parser)

def split_data_label(data):
    """
        将原始含有label和特征的数据split成feature和label两部分
        Args:
            data: pandas dataframe格式
        Rets:
            x_data:
            y_data:
    """
    try:
        y_data = data['label'].values
    except:
        raise ValueError("Label info missing in the data!!!")
    try:
        data.pop("label")
        x_data = data
    except:
        raise ValueError("Label info missing in the data!!!")
    return y_data, x_data 

def parse_args(parser):
    """
        解析输入的参数
        Args:
            parser: argparse.ArgumentParser
        Rets:
            None
    """
    args = parser.parse_args()
    
    #Data type parameter parsing and data loading
    y_train, x_train = None, None
    y_test, x_test = None, None
    y_valid, x_valid = None, None
    
    if args.data_type == "libsvm":
        if args.train_data is not None and len(args.train_data) > 1:
            y_train, x_train = ld.load_libsvm_file(args.train_data[0], args.isDense)
        if args.test_data is not None and len(args.test_data) > 1:
            y_test, x_test = ld.load_libsvm_file(args.test_data[0], args.isDense)
        if args.validation_data is not None and len(args.validation_data) > 1:
            y_valid, x_valid = ld.load_libsvm_file(args.validation_data[0], args.isDense)
    elif args.data_type == "csv_with_schema" or args.data_type == "csv_with_table_header":# 读取的数据类型都是pandas类型
        load_func = ld.load_csv_with_fmap if args.data_type == "csv_with_schema" else ld.load_csv_with_table_header
        if args.train_data is not None:
            train_data = load_func(args.train_data, args.fmap)
            y_train, x_train = split_data_label(train_data)               
        if args.test_data is not None:
            test_data = load_func(args.test_data, args.fmap)
            y_test, x_test = split_data_label(test_data)               
        if args.validation_data is not None:
            valid_data = load_func(args.validation_data, args.fmap)
            y_valid, x_valid = split_data_label(valid_data)               
    else:
        raise ValueError("Wrong data type parameter!")
    
    try:
        boost_round = int(args.boost_round)
    except:
        raise ValueError("Wrong boost round number!!")
    
    try:
        param_dict = eval(args.parameters)
        print colors.YELLOW + "param_dict:", param_dict , colors.ENDC
    except:
        raise ValueError("Wrong parameters!!")
    
    if args.model == "xgboost":
        if args.task == "classification":
            xgb_model = Xgboost.Xgboost.XgbClassifier(params=param_dict, )
            if args.mode == "train":
                xgb_model.train(x_train, y_train, model_saveto=args.model_path) 
            elif args.mode == "predict":
                xgb_model.predict()
            elif args.mode == "analysis":
                xgb_model.analysis(x_test, y_test, model_load_from=args.model_path)


if __name__ == "__main__":
    main()
