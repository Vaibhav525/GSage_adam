import networkx as nx
from collections import defaultdict
import numpy as np


def _split_data(num_nodes, test_split = 3, val_split = 6):
    rand_indices = np.random.permutation(num_nodes)

    test_size = num_nodes // test_split
    val_size = num_nodes // val_split
    train_size = num_nodes - (test_size + val_size)

    test_indexs = rand_indices[:test_size]
    val_indexs = rand_indices[test_size:(test_size+val_size)]
    train_indexs = rand_indices[(test_size+val_size):]
    
    return test_indexs, val_indexs, train_indexs

G :nx.Graph=nx.read_graphml("0.C3Dual.gml")
Bin_type_label=[[1,0,0],[0,1,0],[1,0,0]]	#Type 2=[1,0,0] ,3=[0,1,0],4=[0,0,1]
feat_data = []
labels = [] # label sequence of node
node_map = {} # map node to Node_ID
label_map = {} # map label to Label_ID
i=0
for node in G:
    if(i==1):
        print(node)
        print(type(node))
    features=list(G.nodes[node].values())
    feat_data.append(Bin_type_label[features[0]-2]+[float(x) for x in features[1:]])
    node_map[node] = i
    if not features[0] in label_map:
        label_map[features[0]] = len(label_map)
    labels.append(label_map[features[0]])
    i+=1
feat_data = np.asarray(feat_data)
labels = np.asarray(labels, dtype=np.int64)
adj_lists = defaultdict(set)
for edge in G.edges:
    print(edge[0],edge[1])
    adj_lists[edge[0]].add(edge[1])
    adj_lists[edge[1]].add(edge[0])

assert len(feat_data) == len(labels) == len(adj_lists)
test_indexs, val_indexs, train_indexs = _split_data(feat_data.shape[0],2*feat_data.shape[0],2*feat_data.shape[0])
