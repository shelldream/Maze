# -*- coding:utf-8 -*-
"""
    Description:
        每一维feature对应一个或者多个node,最终这维特征的实际取值是有所有node_weight累乘所得
        
        node的类型包括下面几种:
            NumericFeature: 数值型特征
            EnumFeature: 枚举型特征，取值为0或1
            BinFeature: 每个bin对应0或1
            BinScaleFeature: 每个bin内部做线性的缩放 
        
        条件判断符包括:
            eq: 相等, eq="Nan"表示特征缺失
            gt: 大于
            lt: 小于
            ge: 大于等于
            le: 小于等于
            ne: 不等于
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class Condition(object):
    def __init__(self, operator, value):
        self.operator = operator
        self.value = value
        
    def check(self, check_value):
        if self.operator == "eq":
            if self.value == float("Nan") or check_value is None:
                return True 
            return check_value == self.value
        if self.operator == "gt":
            return check_value > self.value
        if self.operator == "lt":
            return check_value < self.value
        if self.operator == "ge":
            return check_value >= self.value
        if self.operator == "le":
            return check_value <= self.value
        if self.operator == "ne":
            return check_value != self.value
        return false


class Bin(object):
    def __init__(self, feature_name, bin_split_list):
        """
            Args:
                feature_name:
                bin_split_list: list
        """
        self.feature_name = feature_name
        self.bin_split_list = sorted(bin_split_list)
        self.bin_cnt = len(self.bin_split_list) + 1
        if self.bin_cnt == 1:
            raise ValueError("Too less bin split point!!")

    def check(self, check_value, bin_id):
        if bin_id < 0 or bin_id > (self.bin_cnt - 1):
            raise ValueError("Wrong bin id!!")
        if bin_id == (self.bin_cnt - 1):
            if check_value > self.bin_split_list[bin_id-1]:
                return True
            else:
                return False

        if bin_id == 0:
            return True if check_value <= self.bin_split_list[0] else False
        
        if check_value > self.bin_split_list[bin_id-1] and check_value <= self.bin_split_list[bin_id]:
            return True
        else:
            return False


class Node(object):
    def __init__(self, feature_name, node_type, condition):
        self.feature_name = feature_name
        self.node_type = node_type
        self.condition = condition

    def get_value(self, raw_feature_dict, bin_id=None):
        pass

class NumNode(Node):
    def __init__(self, feature_name, node_type, condition):
        Node.__init__(self, feature_name, node_type, condition)
    def get_value(self, raw_feature_dict, bin_id=None):
        return raw_feature_dict[self.feature_name] if self.feature_name in raw_feature_dict else 0.0
         
class EnumNode(Node):
    def __init__(self, feature_name, node_type, condition):
        Node.__init__(self, feature_name, node_type, condition)
        
    def get_value(self, raw_feature_dict, bin_id=None):
        check_value = None if self.feature_name not in raw_feature_dict else raw_feature_dict[self.feature_name]
        return 1.0 if self.condition.check(check_value) else 0.0

class BinNode(Node):
    def __init__(self, feature_name, node_type, condition, bin, bin_id):
        Node.__init__(self, feature_name, node_type, condition)
        self.bin = bin
        self.bin_id = bin_id
        
    def get_value(self, raw_feature_dict):
        if self.feature_name not in raw_feature_dict:
            return 0.0 
        raw_value = raw_feature_dict[self.feature_name]
        if self.bin.check(raw_value, self.bin_id):
            return 1.0
        else:
            return 0.0 


class BinScaleNode(Node):
    def __init__(self, feature_name, node_type, condition, bin_scale_list):
        """
            Args:
                bin_scale_list: a list of tuple (bin_upper, scale_upper)
        """
        Node.__init__(self, feature_name, node_type, condition)
        bin_upper_list = [item[0] for item in bin_scale_list]
        if len(set(bin_upper_list)) != len(bin_upper_list):
            raise ValueError("Duplicated bin upper!!")
        
        self.bin_scale = sorted(bin_scale_list, key=lambda x:x[0])
        self.bin_cnt = len(self.bin_scale) + 1
        if self.bin_cnt == 1:
            raise ValueError("Too less bin split point!!")
        
    def get_value(self, raw_feature_dict, default_min_value=0):
        if self.feature_name not in raw_feature_dict:
            return 0.0
        
        raw_value = raw_feature_dict[self.feature_name]
        for i in xrange(len(self.bin_scale)):
            if 0 == i:
                if raw_value <= self.bin_scale[0][0]:
                    return default_min_value
                else:
                    continue

            if (len(self.bin_scale) - 1) == i:
                return self.bin_scale[len(self.bin_scale)-1][1]
            
            scale_min = self.bin_scale[i-1][1]
            scale_max = self.bin_scale[i][1]
            bin_min = self.bin_scale[i-1][0]
            bin_max = self.bin_scale[i][0]
            
            if raw_value > bin_min and raw_value <= bin_max:
                res = scale_min + (raw_value-bin_min)*(scale_max-scale_min)/(bin_max-bin_min)
                return res
            else:
                continue
        return sef.bin_scale[-1][1]


class Feature(object):
    def __init__(self, index):
        self.index = index
        self.node_list = []

    def append_node(self, fname, ntype, condition, bin_obj=None, bin_scale_list=None, bin_id=None):
        if ntype == "numeric":
            self.node_list.append(NumNode(fname, ntype, condition))
        if ntype == "enum":
            self.node_list.append(EnumNode(fname, ntype, condition)) 
        if ntype == "bin":
            self.node_list.append(BinNode(fname, ntype, condition, bin=bin_obj, bin_id=bin_id))

        if ntype == "bin_scale":
            self.node_list.append(BinScaleNode(fname, ntype, condition, bin_scale_list=bin_scale_list))

    def get_value(self, feature_dict):
        res = 1.0
        for fnode in self.node_list:
            node_weight = fnode.get_value(feature_dict)
            if node_weight == 0:
                return 0.0
            else:
                res *= node_weight
        return res
    
    def get_index(self):
        return self.index

class FeatureGenerator(object):
    def __init__(self, xml_file, black_feature_index=None):
        self.xml_file = xml_file
        self.black_feature_index = black_feature_index if black_feature_index is not None else set()
        self.tree = ET.ElementTree(file=self.xml_file)
        self.quantization_bin_dict = dict()
        self.quantization_bin_scale_dict = dict()
        self.feature_list = []
        self.__parse_xml()

    def __parse_xml(self):
        root = self.tree.getroot()
        for root_child in root:  # parse quantizations first, then parse features
            if "quantizations" == root_child.tag: # parse the quantization part
                for quantization_child in root_child: #for each quantization 
                    attrib_dict = quantization_child.attrib
                    fid = ""
                    tmp_bin_scale_list = []
                    quantization_type = ""
                    for k,v in attrib_dict.items():
                        if 'type' == k:
                            quantization_type = v
                            if 'bin_scale' == v:
                                for bin_scale_child in quantization_child: # for each bin 
                                    bin_scale_dict = bin_scale_child.attrib
                                    tmp_bin_scale_list.append((float(bin_scale_dict["upper"]), float(bin_scale_dict["scale_upper"])))
                            else: #type=bin
                                for bin_child in quantization_child: # for each bin 
                                    bin_dict = bin_child.attrib
                                    tmp_bin_scale_list.append(float(bin_dict['upper']))
                        elif 'id' == k:
                            fid = v
                    if quantization_type == "bin_scale":
                        self.quantization_bin_scale_dict[fid] = tmp_bin_scale_list
                    elif quantization_type == "bin":
                        self.quantization_bin_dict[fid] = tmp_bin_scale_list
        for root_child in root:
            if 'features' == root_child.tag: #parse the features part
                for feature_child in root_child: # for each feature
                    f_index = int(feature_child.attrib["index"])
                    if f_index in self.black_feature_index:
                        continue
                    
                    feature_obj = Feature(f_index)
                    for node_child in feature_child: # for each node
                        node_dict = node_child.attrib
                        fname = node_dict["id"]
                        ntype = node_dict["type"]
                        condition = None
                        bin_obj = None
                        bin_scale_list = None
                        bin_id = None
                         
                        if ntype == "enum":
                            for k,v in node_dict.items():
                                if k != "id" and k!= "type":
                                    condition = Condition(k,float(v))
                        elif ntype == "bin":
                            bin_id = int(node_dict["bin"])
                            bin_obj = Bin(fname, self.quantization_bin_dict[fname])
                        elif ntype == "bin_scale":
                            bin_scale_list=self.quantization_bin_scale_dict[fname]
                        feature_obj.append_node(fname=fname, ntype=ntype, condition=condition, bin_obj=bin_obj, bin_scale_list=bin_scale_list, bin_id=bin_id)
                    self.feature_list.append(feature_obj)
                #print "Load %d feature object totally!"%len(self.feature_list)
                         
    def get_feature_as_sparse_libsvm(self, raw_feature):
        """
            Output feature in sparse libsvm data format.
        """
        output = []
        for feature_obj in self.feature_list:
            index = feature_obj.get_index()
            value = feature_obj.get_value(raw_feature)
            if value != 0:
                output.append("%d:%s"%(index, value))
        return " ".join(output)

    def get_feature_as_dense_libsvm(self, raw_feature):
        pass


if __name__ == "__main__":
    xml_file = "./coupon_gbdt.xml"
    #xml_file = "./demo.xml"
    generator = FeatureGenerator(xml_file)
    #print generator.get_feature_as_sparse_libsvm({"isHit":1, "clicks":1000, "price":2.5, "text_weight":10, "hc_cid3":1315})
    print generator.get_feature_as_sparse_libsvm({})
    #print generator.get_feature_as_sparse_libsvm({"user_sku_click_cnt":1, "user_spu_click_time_delta":0, "COEC":2.323232, "user_spu_click_cnt":0, "sale15":10, "user_spu_click_cnt":0})
