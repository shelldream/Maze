# -*- coding:utf-8 -*-
"""
    Description:
        解析GBDT原始text文件，将GBDT的叶子节点作为特征，生成对应的xml文件
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")

sys.path.append("../")
sys.path.append("./")

import utils.xgboost_utils


class GBDTFeatureExtractor(object):
    def __init__(self, model_file):
        self.model_file = model_file
        self.opt_mapping = {"<":"lt", "<=":"le", ">":"gt", ">=":"ge", "==":"eq", "!=":"ne"}
        self.opt_opposite = {"<":"ge", "<=":"gt", ">":"le", ">=":"lt", "==":"ne", "!=":"eq"}
        
        #feature_list 中每个feature item为一个node_list
        #node_list中每个元素为一个dict，表示一个feature node
        #每个feature node必定包含的内容为: id,type,condition
        self.feature_list = []  
        
        #DFS 遍历每棵树，途径的每个条件存到condition_list中
        #condition_list中每个元素为一个(feature_name, operator, value)
        self.condition_list = []

    def generate_feature_node_list(self):
        all_nodes_info, total_tree_num = utils.xgboost_utils.parse_raw_text_model_file(self.model_file)
        
        def traverse(tree_node_id):
            node_info_dict = all_nodes_info[tree_node_id]
            if "leaf" in node_info_dict:
                print tree_node_id, self.condition_list
                node_list = []
                for cond in self.condition_list:
                    node_dict = {}
                    node_dict["id"] = cond[0]
                    node_dict["type"] = "enum"
                    node_dict["condition"] = cond[1]+"="+'"%s"'%cond[2]
                    node_list.append(node_dict)
                self.feature_list.append(node_list)
            else:
                node_condition = node_info_dict["condition"]
                fname = ""
                opt = ""
                op_opt = ""
                value = ""
                for k,v in self.opt_mapping.items():
                    content = node_condition.split(k)
                    if len(content) == 2:
                        fname = content[0]
                        opt = self.opt_mapping[k]
                        op_opt = self.opt_opposite[k]
                        value = content[1]
                        break
                
                self.condition_list.append((fname, "eq", "nan"))
                traverse(node_info_dict["missing"])
                self.condition_list.pop()
                self.condition_list.append((fname, opt, value))
                traverse(node_info_dict["yes"])
                self.condition_list.pop()
                self.condition_list.append((fname, op_opt, value))
                traverse(node_info_dict["no"])
                self.condition_list.pop()
                
        for tree_id in xrange(total_tree_num):
            traverse("%d_0"%tree_id)    

    def generate_xml_file(self, output_xml_file):
        fw = open(output_xml_file, "w")
        fw.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
        fw.write('<FeatureExtractor version="1.0">\n\n') 
        self.generate_feature_node_list() 
        fw.write("<features>\n") 
        index = 0
        for node_list in self.feature_list:
            fw.write('<feature index="%d">\n'%index)
            index += 1
            last_node_dict = None
            for node_dict in node_list:
                if node_dict == last_node_dict:
                    continue
                node_id = node_dict["id"]
                node_type = node_dict["type"] 
                node_condition = node_dict["condition"]
                last_node_dict = node_dict
                fw.write('\t<node id="%s" type="%s" %s></node>\n'%(node_id, node_type, node_condition))
            fw.write('</feature>\n\n')

        fw.write("</features>\n\n") 
        fw.write("</FeatureExtractor>\n")       
        fw.close()

if __name__ == "__main__":
    model_file = "/home/admin/huangxiaojun/project/Maze/src/model/coupon_model.txt.raw_text"
    xml_file = "./coupon_gbdt.xml"
    gbdt_f_extractor = GBDTFeatureExtractor(model_file)
    gbdt_f_extractor.generate_xml_file(xml_file)
