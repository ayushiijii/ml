import pandas as pd
import math
import numpy as np

# Load the Excel file into a DataFrame
data = pd.read_excel('C:\\Users\\Ayushi\\Desktop\\submissions\\ML\\exp4.xlsx')

# Assuming 'data' has a column called "answer" which is the target variable.
# Modify the condition to remove any irrelevant features
features = [feat for feat in data.columns if feat != "play"]

class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""

def entropy(examples):
    pos = np.sum(examples["play"] == "yes")
    neg = np.sum(examples["play"] == "no")
    
    if pos == 0.0 or neg == 0.0:
        return 0.0
    
    total = pos + neg
    p = pos / total
    n = neg / total
    return -(p * math.log(p, 2) + n * math.log(n, 2))

def info_gain(examples, attr):
    gain = entropy(examples)
    uniq = np.unique(examples[attr])
    
    for u in uniq:
        subdata = examples[examples[attr] == u]
        sub_e = entropy(subdata)
        gain -= (len(subdata) / len(examples)) * sub_e
        
    return gain

def ID3(examples, attrs):
    root = Node()
    max_gain = 0
    max_feat = ""
    
    for feature in attrs:
        gain = info_gain(examples, feature)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
            
    root.value = max_feat
    
    uniq = np.unique(examples[max_feat])
    
    for u in uniq:
        subdata = examples[examples[max_feat] == u]
        
        if entropy(subdata) == 0.0:
            newNode = Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = np.unique(subdata["play"])[0]  # Assuming single class at leaf
            root.children.append(newNode)
        else:
            dummyNode = Node()
            dummyNode.value = u
            new_attrs = attrs.copy()
            new_attrs.remove(max_feat)
            child = ID3(subdata, new_attrs)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    
    return root

def printTree(root: Node, depth=0):
    for _ in range(depth):
        print("\t", end="")
    print(root.value, end="")
    if root.isLeaf:
        print(" -> ", root.pred)
    print()
    for child in root.children:
        printTree(child, depth + 1)

def classify(root: Node, new):
    for child in root.children:
        if child.value == new[root.value]:
            if child.isLeaf:
                print("Predicted Label for new example", new, "is:", child.pred)
                return
            else:
                classify(child.children[0], new)

# Build the decision tree and classify a new example
root = ID3(data, features)
print("Decision Tree is:")
printTree(root)
print("------------------")

# Example new instance for classification
new = {"outlook": "sunny", "temperature": "hot", "humidity": "normal", "wind": "strong"}
classify(root, new)
