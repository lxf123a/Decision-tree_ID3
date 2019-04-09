
# coding: utf-8
import numpy as np


# In[2]:


import pandas as pd
from math import log


# In[3]:


df=pd.read_csv('heart.csv',dtype={'oldpeak':float})


# In[4]:


df


# In[5]:


def entropy(feature):
    '''
    定义熵结算函数，熵的计算公式为enntrop=-sum(p*log(p),p 是概率值
    
    '''
    ##计算概率
    probs=[feature.count(i)/len(feature) for i in set(feature)]
    
    ##计算熵
    entropy=-sum([prob*log(prob,2) for prob in probs])
    
    return entropy
        
    


# In[6]:


entropy(df['target'].tolist())


# In[7]:


def split_dataframe(data,col):
    '''
    基于特征划分数据集,将同值的数据划分块，存储在字典中
    '''
    ##特征的唯一值
    unique_values=data[col].unique()
    ##建立空字典
    
    result_dict={elem :pd.DataFrame for elem in unique_values}
    for key in result_dict.keys():
        result_dict[key]=data[:][data[col]==key]
    return result_dict
    


# In[8]:


split_dataframe(df,'age')


# In[9]:


def choose_best_col(df,label):
    '''
    根据熵、划分的数据集计算信息增益来则最佳的特征
    
    '''
    ##计算目标熵
    entropy_D=entropy(df[label].tolist())
    ###存储特征名称（列名）
    cols=[col for col in df.columns if col not in [label]]
    ##初始化
    max_value,best_col=-999,None
    max_splited=None
    
    for col in  cols:
        splited_set=split_dataframe(df,col)
        entropy_DA=0
        for subset_col,subset in splited_set.items():
            ###计算特征划分后的熵
            entropy_Di=entropy(subset[label].tolist())
            ###计算当前特征的熵（权重累加）
            entropy_DA+=len(subset)/len(df)*entropy_Di
        info_gain=entropy_D-entropy_DA
        if info_gain>max_value:
            max_value,best_col=info_gain,col
            max_splited=splited_set
    return max_value,best_col,max_splited
    
    


# In[129]:


choose_best_col(df,'target')


# In[10]:


class Tree:
    class Node:
        def __init__(self,name):
            self.name=name
            self.connections={}
#             self.rightchild=rname
#             self.leftname=lname
        def connect(self,label,node):
            self.connections[label]=node
    def __init__(self,data,label):
        self.columns=data.columns
        self.data=data
        self.label=label
        self.root=self.Node('root')
        
    ###打印树
    def print_tree(self,node,tabs):
        print(tabs+str(node.name))

        for connection,child_node in node.connections.items():
            print(tabs + "\\" + "(" + str(connection) + ")")
            self.print_tree(child_node, tabs + "\\")    
    ###类开始
    def construct_tree(self):
        self.construct(self.root,"",self.data,self.columns)
    ###递归构建决策树
    def construct(self,parent_node,parent_connection_label,input_data,columns):
        '''
        input:self.label:目标列名
        columns:t特征名称
        '''
        max_value,best_col,max_splited=choose_best_col(input_data[columns],self.label)
        print(best_col)
        
        if not best_col:
            print('^^^^^^^^^^^^^^^^^^^^^^')
            node=self.Node(input_data[self.label].iloc[0])
            parent_node.connect(parent_connection_label,node)
            return
        ###选择的信息增益最大的特征
        
        node=self.Node(best_col)
        
        parent_node.connect(parent_connection_label,node)
#         print(parent_node.connections)
#         print(node)
        new_columns=[col for col in columns if col !=best_col]
        
        for split_value,split_data in max_splited.items():
            self.construct(node,split_value,split_data,new_columns)
        
    
        


# In[11]:


tree1=Tree(df,'target')
tree1.construct_tree()
print(tree1)
tree1.print_tree(tree1.root,"")

