import sys
import os
import torch
import pandas as pd
import argparse
import pyhocon
import random
from collections import defaultdict


from src.dataCenter import *
from src.utils import *
from src.models import *

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')
parser.add_argument('--dataSet', type=str, default='NormLJ')
parser.add_argument('--NGraphs', type=int, default=10)
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--config', type=str, default='./src/experiments.conf')
args = parser.parse_args()

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load config file
    config = pyhocon.ConfigFactory.parse_file(args.config)
    # load data
    ds = args.dataSet
    ds_Val=args.Val
    ###Put in loop pass to apply model
    dataCenter = DataCenter(config)
    dataCenter.load_dataSet(ds)
    features = torch.FloatTensor(getattr(dataCenter, ds+'_feats')[0]).to(device)

    dataCenter_val = DataCenter(config)
    dataCenter_val.load_dataSet(ds+"_val")
    features_val = torch.FloatTensor(getattr(dataCenter_val, ds+"_val"+'_feats')[0]).to(device)
    ##########
    graphSage = torch.load('models/Final_model.torch', map_location=torch.device('cpu'))
    graphSage.eval()
    num_labels = len(set(getattr(dataCenter, ds+'_labels')[0]))
    classification = Classification(config['setting.hidden_emb_size'], num_labels)
    classification.to(device)

    unsupervised_loss = [UnsupervisedLoss(getattr(dataCenter, ds+'_adj_lists')[i], getattr(dataCenter, ds+'_train')[i], device) for i in range(0,config['setting.N_graph_train_index_end']-config['setting.N_graph_train_index_start'],1)]
    unsupervised_loss_val = [UnsupervisedLoss(getattr(dataCenter_val, ds+"_val"+'_adj_lists')[i], getattr(dataCenter_val, ds+"_val"+'_train')[i], device) for i in range(0,config['setting.N_graph_val_index_end']-config['setting.N_graph_val_index_start'],1)]

    if args.learn_method == 'sup':
        print('GraphSage with Supervised Learning')
    elif args.learn_method == 'plus_unsup':
        print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
    else:
        print('GraphSage with Net Unsupervised Learning')
    Best_loss=100000000

    Embs_graph=dict()
    Link_samples=dict() #Format: Link_samples[Graph_id]=(Positive Samples list,Negative Samples list)
    for Gid in range(args.NGraphs):
        nodes = getattr(dataCenter_val, ds+"_val"+'_train')[Gid]
        features = torch.FloatTensor(getattr(dataCenter_val, ds+"_val"+'_feats')[Gid]).to(device)
        adj_list=getattr(dataCenter_val, ds+"_val"+'_adj_lists')[Gid]
        
        #modify adj_list: remove 10% positive samples, 10% neg samps
        Positive_links=[]
        Negative_links=[]
        for i in range(400):
            #--------Negative links-----------
            
            #Randomly choose a node
            e1=random.choice(nodes)
            found=False
            while(found==False):
                #Chose a node not in adj_list of e1
                e2=random.choice(nodes)
                if e2 not in adj_list[e1]:
                    found=True
                    #When found add to Negative links
                    Neg_link=(e1,e2)
                    Negative_links+=[Neg_link]
                    

            #------Positive links-----
            #Randomly choose a link
            n1=random.choice(nodes)
            n2=random.choice(adj_list[n1])
            #Add to positive example
            Pos_link=(n1,n2)
            Positive_links+=[Pos_link]
            #Remove link from adj_list
            adj_list[n1].remove(n2)
            adj_list[n2].remove(n1)            
        
        Embs=graphSage(np.asarray(nodes),features,adj_list)
        #Collect Embs in a list , keep track of graph and embs
        Embs_graph[Gid]=Embs
        Link_samples[Gid]=Embs
        # Extra_node_data=getattr(dataCenter_val, ds+"_val"+'_Ext')[Gid]
        # t_np = Final_embs.detach().cpu().numpy() #convert to Numpy array
        # df = pd.DataFrame(t_np) #convert to a dataframe
        # df1 = pd.DataFrame(Extra_node_data) 
        # df.to_csv("Fin_embs_"+str(config['setting.N_graph_val_index_start']+Gid)+".csv",index=False) #save to file
        # df1.to_csv("Fin_Ext_"+str(config['setting.N_graph_val_index_start']+Gid)+".csv",index=False) #save to file
        



