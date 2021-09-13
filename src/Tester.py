import networkx as nx
import pyhocon
import pandas as pd
import numpy as np
from collections import defaultdict


def _split_data(self, num_nodes, test_split = 3, val_split = 6):
    rand_indices = np.random.permutation(num_nodes)

    test_size = num_nodes // test_split
    val_size = num_nodes // val_split
    train_size = num_nodes - (test_size + val_size)

    test_indexs = rand_indices[:test_size]
    val_indexs = rand_indices[test_size:(test_size+val_size)]
    train_indexs = rand_indices[(test_size+val_size):]
    
    return test_indexs, val_indexs, train_indexs

config = pyhocon.ConfigFactory.parse_file('./src/experiments.conf')
	
NormLJ = config['file_path.NormLJ']
test_indexs_i=[]
val_indexs_i=[]
train_indexs_i=[]
feat_data_i=[]
labels_i=[]
adj_lists_i=[]
Ext_data_i=[]
flag=True
for k in range(config['setting.N_graph_val_index_start'],config['setting.N_graph_val_index_end'],1):
    
    G :nx.Graph=nx.read_graphml(NormLJ+"/"+str(k)+"Norm.gml")
    Bin_type_label=[[1,0],[0,1]]
    feat_data = []
    Ext_data=[]
    labels = [] # label sequence of node
    node_map = {} # map node to Node_ID
    label_map = {} # map label to Label_ID
    i=1
    for node in G:
        features=list(G.nodes[node].values())
        feat_data.append(Bin_type_label[features[0]-1]+[float(x) for x in features[1:4]])
        Ext_data.append([float(x) for x in features[5:]])
        if(flag==True):
            print(feat_data)
            flag=False
        node_map[node] = i
        if not features[0] in label_map:
            label_map[features[0]] = len(label_map)
        labels.append(label_map[features[0]])
        i+=1
    
   
    feat_data = np.asarray(feat_data)
    Ext_data=np.asarray(Ext_data)
    labels = np.asarray(labels, dtype=np.int64)
    adj_lists = defaultdict(set)
    for edge in G.edges:
        e1=node_map[edge[0]]
        e2=node_map[edge[1]]
        adj_lists[e1].add(e2)
        adj_lists[e2].add(e1)
        
    assert len(feat_data) == len(labels) == len(adj_lists)
    test_indexs, val_indexs, train_indexs = _split_data(feat_data.shape[0],2*feat_data.shape[0],2*feat_data.shape[0])
    test_indexs_i+=[test_indexs]
    val_indexs_i+=[val_indexs]
    train_indexs_i+=[train_indexs]
    feat_data_i+=[feat_data]
    Ext_data_i+=[Ext_data]
    labels_i+=[labels]
    adj_lists_i+=[adj_lists]

    Ext_data=np.asarray(Ext_data)
    a=pd.DataFrame(Ext_data)
    a.to_csv("Ext.csv",index=False)

