import sys
import os
import torch
import pandas as pd
import argparse
import pyhocon
import random
from IPython.display import display, HTML
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from src.dataCenter import *
from src.utils import *
from src.models import *
from src.Link_Prediction import *

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')
parser.add_argument('--dataSet', type=str, default='NormLJ')
parser.add_argument('--NGraphs', type=int, default=10)
parser.add_argument('--seed', type=int, default=824)
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
    ###Put in loop pass to apply model
    
    dataCenter = DataCenter(config)
    dataCenter.load_dataSet(ds+"_val")
    features= torch.FloatTensor(getattr(dataCenter, ds+"_val"+'_feats')[0]).to(device)
    ##########
    graphSage = torch.load('models/Final_model.torch', map_location=torch.device('cpu'))
    graphSage.eval()
    
    def graphsage_embedding(graph_nodes,node_features,adj_list,name):
        print(f"Forward passing GraphSAGE for '{name}':")
        Embs=graphSage(graph_nodes,node_features,adj_list).detach().cpu().numpy() 
        def get_embedding(u):
            return Embs[u]
        return get_embedding
    

    unsupervised_loss_val = [UnsupervisedLoss(getattr(dataCenter, ds+"_val"+'_adj_lists')[i], getattr(dataCenter, ds+"_val"+'_train')[i], device) for i in range(0,config['setting.N_graph_val_index_end']-config['setting.N_graph_val_index_start'],1)]
   
    for Gid in range(args.NGraphs):
        nodes = getattr(dataCenter, ds+"_val"+'_train')[Gid]
        features = torch.FloatTensor(getattr(dataCenter, ds+"_val"+'_feats')[Gid]).to(device)
        adj_list=getattr(dataCenter, ds+"_val"+'_adj_lists')[Gid]
        myexamples=[]     #Set of edges 
        mylabels=[]       #label of edge :0:negative, 1:positive
        
        #modify adj_list: remove 10% positive samples, 10% neg samps ~400 each
        
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
                    Neg_link=[e1,e2]
                    myexamples+=[Neg_link]
                    mylabels+=[0]
                    

            #------Positive links-----
            #Randomly choose a link
            n1=random.choice(nodes)
            n2=random.choice(list(adj_list[n1]))
            #Add to positive example
            Pos_link=[n1,n2]
            myexamples+=[Pos_link]
            mylabels+=[1]
            #Remove link from adj_list
            adj_list[n1].remove(n2)
            adj_list[n2].remove(n1)            
        
        examples=np.array(myexamples)
        labels=np.array(mylabels)
        
       
        (
            examples_,
            examples_model_selection,
            labels_,
            labels_model_selection,
        ) = train_test_split(examples, labels, train_size=0.8, test_size=0.2)

        (
            examples_train,
            examples_test,
            labels_train,
            labels_test,
        ) = train_test_split(examples_, labels_, train_size=0.8, test_size=0.2)
        
        def run_link_prediction(binary_operator,embedding_train):
            clf = train_link_prediction_model(
                examples_train, labels_train, embedding_train, binary_operator
            )
            score = evaluate_link_prediction_model(
                clf,
                examples_model_selection,
                labels_model_selection,
                embedding_train,
                binary_operator,
            )

            return {
                "classifier": clf,
                "binary_operator": binary_operator,
                "score": score,
            }


        binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]


        def train_and_evaluate(embedding, name):

            embedding_train = embedding(nodes,features,adj_list, "Train Graph")

            # Train the link classification model with the learned embedding
            results = [run_link_prediction(op, embedding_train) for op in binary_operators]
            best_result = max(results, key=lambda result: result["score"])
            print(
                f"\nBest result with '{name}' embeddings from '{best_result['binary_operator'].__name__}'"
            )
            display(
                pd.DataFrame(
                    [(result["binary_operator"].__name__, result["score"]) for result in results],
                    columns=("name", "ROC AUC"),
                ).set_index("name")
            )

            # Evaluate the best model using the test set
            test_score = evaluate_link_prediction_model(
                best_result["classifier"],
                examples_test,
                labels_test,
                embedding_train,
                best_result["binary_operator"],
            )

            return test_score

        graphsage_result = train_and_evaluate(graphsage_embedding, "Graph_"+str(Gid))

