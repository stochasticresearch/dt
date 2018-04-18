#!/usr/bin/env python

# adapted from:
# https://gist.githubusercontent.com/darden1/9f99d104867503204313064c594f2f69/raw/bf170a13e1e4c63208cb5f7bad0fab5641240e4a/my_decision_tree.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pydotplus
    
import matlab.engine

class Node(object):
    def __init__(self, criterion, criterion_transform='Matlab', criterion_additional_args=[],
        criterion_minnumsamps=1,
        max_depth=None, random_state=None):
        self.criterion    = criterion
        self.criterion_transform = criterion_transform
        self.criterion_additional_args = criterion_additional_args
        self.criterion_minnumsamps = criterion_minnumsamps
        self.max_depth    = max_depth
        self.random_state = random_state
        self.depth        = None
        self.left         = None
        self.right        = None
        self.feature      = None
        self.threshold    = None
        self.label        = None
        #self.impurity     = None
        self.info_gain    = None
        self.num_samples  = None
        self.num_classes  = None
        
    def split_node(self, sample, target, depth, init_num_classes):
        self.depth = depth

        self.num_samples = len(target)
        self.num_classes = [len(target[target==i]) for i in init_num_classes]

        if len(np.unique(target)) == 1:
            self.label = target[0]
            #self.impurity = self.criterion_func(target)
            return
        class_count = {i: len(target[target==i]) for i in np.unique(target)}
        self.label = max(class_count.items(), key=lambda x:x[1])[0]
        #self.impurity = self.criterion_func(target)
        
        num_features = sample.shape[1]
        self.info_gain = 0.0

        if self.random_state!=None:
            np.random.seed(self.random_state)
        f_loop_order = np.random.permutation(num_features).tolist()
        for f in f_loop_order:
            uniq_feature = np.unique(sample[:, f])
            split_points = (uniq_feature[:-1] + uniq_feature[1:]) / 2.0

            for threshold in split_points:
                target_l = target[sample[:, f] <= threshold] 
                target_r = target[sample[:, f] >  threshold]

                x = sample[:,f]
                y = target
                x_l = sample[sample[:, f] <= threshold,f]
                y_l = target_l
                x_r = sample[sample[:, f] >  threshold,f]
                y_r = target_r

                if(len(x)<self.criterion_minnumsamps or len(y)<self.criterion_minnumsamps or 
                    len(x_l)<self.criterion_minnumsamps or len(y_l)<self.criterion_minnumsamps or 
                    len(x_r)<self.criterion_minnumsamps or len(y_r)<self.criterion_minnumsamps):
                    continue
                val = self.calc_info_gain(x,y,x_l,y_l,x_r,y_r)
                if self.info_gain < val:
                    self.info_gain = val
                    self.feature   = f
                    self.threshold = threshold

        if self.info_gain == 0.0:
            return
        if depth == self.max_depth:
            return

        sample_l   = sample[sample[:, self.feature] <= self.threshold]
        target_l   = target[sample[:, self.feature] <= self.threshold]
        self.left  = Node(self.criterion, 
            self.criterion_transform, self.criterion_additional_args, 
            self.criterion_minnumsamps,
            self.max_depth)
        self.left.split_node(sample_l, target_l, depth + 1, init_num_classes)

        sample_r   = sample[sample[:, self.feature] > self.threshold]
        target_r   = target[sample[:, self.feature] > self.threshold]
        self.right = Node(self.criterion, 
            self.criterion_transform, self.criterion_additional_args, 
            self.criterion_minnumsamps,
            self.max_depth)
        self.right.split_node(sample_r, target_r, depth + 1, init_num_classes)

    def calc_info_gain(self, x, y, x_l, y_l, x_r, y_r):
        if(self.criterion_transform=='Matlab'):
            xx   = convert_np_to_matlab_and_transpose(x)
            yy   = convert_np_to_matlab_and_transpose(y)
            xx_l = convert_np_to_matlab_and_transpose(x_l)
            yy_l = convert_np_to_matlab_and_transpose(y_l)
            xx_r = convert_np_to_matlab_and_transpose(x_r)
            yy_r = convert_np_to_matlab_and_transpose(y_r)
            
        else:
            xx = x
            yy = y
            xx_l = x_l
            yy_l = y_l
            xx_r = x_r
            yy_r = y_r

        cri_p  = self.criterion(xx,yy,*self.criterion_additional_args)
        cri_cl = self.criterion(xx_l,yy_l,*self.criterion_additional_args)
        cri_cr = self.criterion(xx_r,yy_r,*self.criterion_additional_args)
        
        return cri_p - len(y_l)/float(len(y))*cri_cl - len(y_r)/float(len(y))*cri_cr

    def predict(self, sample):
        if self.feature == None or self.depth == self.max_depth:
            return self.label
        else:
            if sample[self.feature] <= self.threshold:
                return self.left.predict(sample)
            else:
                return self.right.predict(sample)

def convert_np_to_matlab_and_transpose(x):
    xx = matlab.double(x.tolist())
    xx.reshape((xx.size[1],xx.size[0]))
    return xx
    
class TreeAnalysis(object):
    def __init__(self):
        self.num_features = None
        self.importances  = None
        
    def compute_feature_importances(self, node):
        if node.feature == None:
            return
        
        self.importances[node.feature] += node.info_gain*node.num_samples
        
        self.compute_feature_importances(node.left)
        self.compute_feature_importances(node.right)
        
    def get_feature_importances(self, node, num_features, normalize=True):
        self.num_features = num_features
        self.importances  = np.zeros(num_features)
        
        self.compute_feature_importances(node)
        self.importances /= node.num_samples

        if normalize:
            normalizer = np.sum(self.importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                self.importances /= normalizer
        return self.importances
    
            
class DecisionTree(object):
    def __init__(self, criterion, 
        criterion_transform='Matlab', criterion_additional_args=[],
        criterion_minnumsamps=1,
        max_depth=None, random_state=None):
        
        self.tree          = None
        self.criterion     = criterion
        self.criterion_transform = criterion_transform
        self.criterion_additional_args = criterion_additional_args
        self.criterion_minnumsamps = criterion_minnumsamps
        self.max_depth     = max_depth
        self.random_state  = random_state
        self.tree_analysis = TreeAnalysis()

    def fit(self, sample, target):
        self.tree = Node(self.criterion, 
            self.criterion_transform, self.criterion_additional_args,
            self.criterion_minnumsamps, 
            self.max_depth, self.random_state)
        self.tree.split_node(sample, target, 0, np.unique(target))
        self.feature_importances_ = self.tree_analysis.get_feature_importances(self.tree, sample.shape[1])

    def predict(self, sample):
        pred = []
        for s in sample:
            pred.append(self.tree.predict(s))
        return np.array(pred)

    def score(self, sample, target):
        return sum(self.predict(sample) == target)/float(len(target))
    
    
class TreeStructure(object):
    def __init__(self):
        self.num_node = None
        self.dot_data = None
        
    def print_tree(self, node, feature_names, class_names, parent_node_num):
        node.my_node_num = self.num_node
        node.parent_node_num = parent_node_num

        tree_str = ""
        if node.feature == None or node.depth == node.max_depth:
            tree_str += str(self.num_node) + " [label=<" + node.criterion + " = " + "%.4f" % (node.impurity) + "<br/>" \
                                           + "samples = " + str(node.num_samples) + "<br/>" \
                                           + "value = " + str(node.num_classes) + "<br/>" \
                                           + "class = " + class_names[node.label] + ">, fillcolor=\"#00000000\"] ;\n"
            if node.my_node_num!=node.parent_node_num:
                tree_str += str(node.parent_node_num) + " -> " 
                tree_str += str(node.my_node_num)
                if node.parent_node_num==0 and node.my_node_num==1:
                    tree_str += " [labeldistance=2.5, labelangle=45, headlabel=\"True\"] ;\n"
                elif node.parent_node_num==0:
                    tree_str += " [labeldistance=2.5, labelangle=-45, headlabel=\"False\"] ;\n"
                else:
                    tree_str += " ;\n"
            self.dot_data += tree_str
        else:
            tree_str += str(self.num_node) + " [label=<" + feature_names[node.feature] + " &le; " + str(node.threshold) + "<br/>" \
                                           + node.criterion + " = " + "%.4f" % (node.impurity) + "<br/>" \
                                           + "samples = " + str(node.num_samples) + "<br/>" \
                                           + "value = " + str(node.num_classes) + "<br/>" \
                                           + "class = " + class_names[node.label] + ">, fillcolor=\"#00000000\"] ;\n"
            if node.my_node_num!=node.parent_node_num:
                tree_str += str(node.parent_node_num) + " -> " 
                tree_str += str(node.my_node_num)
                if node.parent_node_num==0 and node.my_node_num==1:
                    tree_str += " [labeldistance=2.5, labelangle=45, headlabel=\"True\"] ;\n"
                elif node.parent_node_num==0:
                    tree_str += " [labeldistance=2.5, labelangle=-45, headlabel=\"False\"] ;\n"
                else:
                    tree_str += " ;\n"
            self.dot_data += tree_str

            self.num_node+=1
            self.print_tree(node.left, feature_names, class_names, node.my_node_num)
            self.num_node+=1
            self.print_tree(node.right, feature_names, class_names, node.my_node_num)

    def export_graphviz(self, node, feature_names, class_names):
        self.num_node = 0
        self.dot_data = "digraph Tree {\nnode [shape=box, style=\"filled, rounded\", color=\"black\", fontname=helvetica] ;\nedge [fontname=helvetica] ;\n"
        self.print_tree(node, feature_names, class_names, 0)
        self.dot_data += "}"
        return self.dot_data
        
if __name__ == "__main__":
    main()