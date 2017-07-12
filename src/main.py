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
    parser.add_argument("--parameters", default="{}", dest="parameters", help="Choose the parameters for your model and \
        the format of your parameters is dict format!")
    
    parse_args(parser)


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
    elif args.data_type == "csv_with_schema":
        if args.train_data is not None:
            x_train = ld.load_csv_with_fmap(args.train_data, args.fmap)
            try:
                y_train = x_train['label'].values
            except:
                raise ValueError("Label info missing in the training data!!!")
        if args.test_data is not None:
            x_test = ld.load_csv_with_fmap(args.test_data, args.fmap)
        if args.validation_data is not None:
            x_valid = ld.load_csv_with_fmap(args.validation_data, args.fmap)
    elif args.data_type == "csv_with_table_header":
        pass
    else:
        raise ValueError("Wrong data type parameter!")
    
    try:
        param_dict = eval(args.parameters)
        print colors.YELLOW + "param_dict:", param_dict , colors.ENDC
    except:
        raise ValueError("Wrong parameters!!")
    
    if args.model == "xgboost":
        """ 如果使用Xgboost，保证训练数据
        """
        if x_train is not None:
            if y_train is not None:
                train_data = xgb.DMatrix(x_train, y_train)
            else:
                train_data = xgb.DMatrix(x_train)

        if x_test is not None:
            if y_test is not None:
                test_data = xgb.DMatrix(x_test, y_test)
            else:
                test_data = xgb.DMatrix(x_test)
        
        if x_valid is not None:
            if y_valid is not None:
                valid_data = xgb.DMatrix(x_valid, y_valid)
            else:
                valid_data = xgb.DMatrix(x_valid)
        
        if args.task == "classification":
            Xgb = Xgboost.Xgboost.XgbClassifier(train_data=train_data, params=param_dict)
            if args.mode == "train":
                Xgb.train() 

if __name__ == "__main__":
    main()
