# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")
import re

def parse_raw_text_model_file(filename):
    "解析xgboost 原始的text文件"
    all_nodes_info = dict()

    tree_id = None
    total_tree_num = 0

    with open(filename, "r") as fr:
        for line in fr:
            line = line.rstrip()
            root_match_res = re.match(re.compile(r"booster\[\d+\]"), line) #find the root node of a single tree
            if root_match_res is not None:
                tree_id = root_match_res.group().lstrip("booster[").rstrip("]")
                total_tree_num += 1
                 
            tree_node_match_res = re.search(re.compile(r"\d+:"), line)
            if tree_node_match_res is not None:
                node_id = tree_id + "_" + tree_node_match_res.group().rstrip(":")
                all_nodes_info[node_id] = dict()
            
                yes_match_res = re.search(re.compile("yes=\d+"), line)
                no_match_res = re.search(re.compile(r"no=\d+"), line)
                missing_match_res = re.search(re.compile(r"missing=\d+"), line)
                
                if yes_match_res is not None and no_match_res is not None and missing_match_res is not None:
                    condition_match_res = re.search(re.compile(r"\[.*\]"), line)
                    condition = condition_match_res.group().lstrip("[").rstrip("]")
                    yes_node_id = tree_id + "_" + re.search(re.compile(r"yes=\d+"), line).group().lstrip("yes=")
                    no_node_id = tree_id + "_" + re.search(re.compile(r"no=\d+"), line).group().lstrip("no=")
                    missing_node_id = tree_id + "_" + re.search(re.compile(r"missing=\d+"), line).group().lstrip("missing=")
            
                    all_nodes_info[node_id]["condition"] = condition
                    all_nodes_info[node_id]["yes"] = yes_node_id
                    all_nodes_info[node_id]["no"] = no_node_id
                    all_nodes_info[node_id]["missing"] = missing_node_id

                leaf_match_res = re.search(re.compile("leaf=.*"), line)
                
                if leaf_match_res is not None:
                    leaf_value = float(leaf_match_res.group().lstrip("leaf=") )
                    all_nodes_info[node_id]["leaf"] = leaf_value
    return all_nodes_info, total_tree_num


def get_xgboost_score(all_nodes_info, total_tree_num, feature_dict):
    score = 0.0
    node_id = None
    operators = ["<", "<=", ">", ">=", "=="]
    for i in xrange(total_tree_num):
        tree_id = str(i)
        node_id = tree_id + "_0"
        while True: #traverse a singe tree
            if "condition" in all_nodes_info[node_id]:
                for operator in operators:
                    try:
                        condition_key, condition_value = all_nodes_info[node_id]["condition"].split(operator)
                        if condition_key not in feature_dict:
                            print "feature %s is missing in your feature dict. Set the value of the feature nan!"%condition_key
                            real_value = None
                        else:
                            real_value = feature_dict[condition_key]
                        try:
                            condition_value = float(condition_value)
                        except:
                            condition_value = int(condition_value)
                        
                        if operator == "<":
                            if real_value is None:
                                node_id = all_nodes_info[node_id]["missing"]
                            elif real_value < condition_value:
                                node_id = all_nodes_info[node_id]["yes"]
                            else:
                                node_id = all_nodes_info[node_id]["no"]
                        break
                    except:
                        continue
            elif "leaf" in all_nodes_info[node_id]:
                score += all_nodes_info[node_id]["leaf"]
                break  #Break the loop until reach the leaf node
    return score


def tunning():
    """
        Usage:
            python xgboost_tunning.py demo_model.txt
            please input feature dict or modify your feature:{"bm25_2_all_dump":5.32322, "bm25_12_all_dump":10.3232, "bm25_12_dump":3.322, "bm25_2_dump":0.3232}
            score:3.271109
            please input feature dict or modify your feature:"bm25_2_all_dump":0      
            Now the feature dict is  {'bm25_2_dump': 0.3232, 'bm25_12_dump': 3.322, 'bm25_12_all_dump': 10.3232, 'bm25_2_all_dump': 0.0}
            score:3.450927
            please input feature dict or modify your feature:"bm25_2_all_dump":0,'bm25_12_all_dump': 0     
            Now the feature dict is  {'bm25_2_dump': 0.3232, 'bm25_12_dump': 3.322, 'bm25_12_all_dump': 0.0, 'bm25_2_all_dump': 0.0}
            score:2.450927
    """
    filename = sys.argv[1]
    all_nodes_info, total_tree_num = parse_raw_text_model_file(filename)
    feature_dict = dict()
    while True:
        raw_data = raw_input("please input feature dict or modify your feature:")
        try:
            feature_dict = eval(raw_data)
            xgboost_score = get_xgboost_score(all_nodes_info, total_tree_num, feature_dict)
            print "score:%f"%xgboost_score
        except:
            try:
                feature_value_list = raw_data.split(",")
                for item in feature_value_list:
                    feature, value = item.split(":")
                    feature = feature.strip('"').strip("'")
                    feature_dict[feature] = float(value)
                print "Now the feature dict is ", feature_dict
                xgboost_score = get_xgboost_score(all_nodes_info, total_tree_num, feature_dict)
                print "score:%f"%xgboost_score
            except:
                print "Wrong feature dict format!!"
                continue
        

if __name__ == "__main__":
    tunning()
