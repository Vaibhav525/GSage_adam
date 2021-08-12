import networkx as nx
import numpy as np
from collections import defaultdict

G :nx.Graph=nx.read_graphml("md.0.lammpstrj.Graph.gml")

def _split_data(num_nodes, test_split = 3, val_split = 6):
    rand_indices = np.random.permutation(num_nodes)

    test_size = num_nodes // test_split
    val_size = num_nodes // val_split
    train_size = num_nodes - (test_size + val_size)

    test_indexs = rand_indices[:test_size]
    val_indexs = rand_indices[test_size:(test_size+val_size)]
    train_indexs = rand_indices[(test_size+val_size):]
    
    return test_indexs, val_indexs, train_indexs

Bin_type_label=[[1,0],[0,1]]
feat_data = []
labels = [] # label sequence of node
node_map = {} # map node to Node_ID
label_map = {} # map label to Label_ID
i=0
for node in G:
    features=list(G.nodes[node].values())
    feat_data.append(Bin_type_label[features[0]-1]+[float(x) for x in features[1:]])
    node_map[int(node)] = i
    if not features[0] in label_map:
        label_map[features[0]] = len(label_map)
    labels.append(label_map[features[0]])
    i+=1
feat_data = np.asarray(feat_data)
labels = np.asarray(labels, dtype=np.int64)
adj_lists = defaultdict(set)

for edge in G.edges:
    adj_lists[edge[0]].add(edge[1])
    adj_lists[edge[1]].add(edge[0])

print(len(feat_data),len(labels),len(adj_lists))
assert len(feat_data) == len(labels) == len(adj_lists)
test_indexs, val_indexs, train_indexs = _split_data(feat_data.shape[0],2*feat_data.shape[0],2*feat_data.shape[0])

print(feat_data,node_map,labels,test_indexs,val_indexs,train_indexs)