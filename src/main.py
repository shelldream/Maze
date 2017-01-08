#-*- coding:utf-8 -*-
"""
    Description: main.py
    Author: shelldream
    Date: 2017-01-07
"""
import sys
reload(sys).setdefaultencoding('utf-8')
import argparse
import Xgboost.Xgboost

from utils import load_txt_file

def main():
    parser = argparse.ArgumentParser(description="This is Maze platform!")

    #model type
    parser.add_argument("--model", choices=["xgboost", ], dest="model", default="xgboost", help="Choose the model you want to use.\
    	The choices include xgboost.")

    #task type
    parser.add_argument("--task", choices=["regression", "classification", "ranking"], dest="task", default="regression",
    	help="Choose the task type you want to use! The choices include regression, classification, ranking")

    #task mode
    parser.add_argument("--mode", choices=["train", "predict", "analysis"], dest="mode", default="train")

    #data path
    parser.add_argument("--train_data_path", action="append", dest="train_data", default=[], help="Choose the \
    	training data path. You can choose more than one file path.")
    parser.add_argument("--test_data_path", action="append", dest="test_data", default=[], help="Choose the \
    	test data path. You can choose more than one file path.")

    #parameter
    parser.add_argument("--parameters", default="{}", dest="parameters", help="Choose the parameters for your model and \
    	the format of your parameters is a dict!")

    args = parser.parse_args()
    try:
    	param_dict = eval(args.parameters)
    except:
    	raise ValueError("Wrong parameters!!")
    
    if args.model == "xgboost":
    	if args.task == "classification":
    		Xgb = Xgboost.Xgboost.XgboostClassifier()


if __name__ == "__main__":
    main()
