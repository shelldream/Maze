#-*- coding:utf-8 -*-
"""
    Description: 
        main.py
        支持的原始的数据文本格式：
            1. libsvm格式
            2. CSV
    Author: shelldream
    Date: 2017-01-07
"""
import sys
reload(sys).setdefaultencoding('utf-8')
import argparse
import Xgboost.Xgboost

from utils import load_libsvm_txt_file


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
    parser.add_argument("--data_type", choices=["libsvm", "csv_with_schema", "csv_without_schema"], dest="data_type", default="libsvm", help="Choose the data type you want to use.\
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
        y_train, x_train = load_libsvm_txt_file(args.isDense, args.train_data)
        if args.test_data is not None:
            y_test, x_test = load_libsvm_txt_file(args.isDense, args.test_data)
        if args.validation_data is not None:
            y_valid, x_valid = load_libsvm_txt_file(args.isDense, args.validation_data)
    elif args.data_type == "csv_with_schema":
        pass
    elif args.data_type == "csv_without_schema":
        pass
    else:
        raise ValueError("Wrong data type parameter!")
    
    try:
        param_dict = eval(args.parameters)
    except:
        raise ValueError("Wrong parameters!!")
    
    if args.model == "xgboost":
        """ 如果使用Xgboost，保证训练数据
        """
        if args.task == "classification":
            Xgb = Xgboost.Xgboost.XgbClassifier(train_data=)

    

if __name__ == "__main__":
    main()
