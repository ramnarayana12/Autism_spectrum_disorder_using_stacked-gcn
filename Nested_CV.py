
import os
import time
import argparse
import numpy as np
import networkx as nx
import wandb
import seaborn as sns
import torch
import torch.optim as optim
import pandas as pd
from model import GCN
from deepwalk import deepWalk
import matplotlib.pyplot as plt

from metrics import torchmetrics_accuracy, torchmetrics_auc, correct_num, prf

from dataloader import dataloader
from model import HybridModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc,confusion_matrix,roc_curve
from metrics import plot_confusion_matrix



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=250, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--atlas', default='ho', help='atlas for network construction (node definition) default: aal, see preprocessed-connectomes-project.org/abide/Pipelines.html for more options ')
parser.add_argument('--num_features', default=2000, type=int, help='Number of features to keep for the feature selection step (default: 2000)')
parser.add_argument('--folds', default=10, type=int, help='For cross validation, specifies which fold will be used. All folds are used if set to 11 (default: 11)')
parser.add_argument('--connectivity', default='correlation', help='Type of connectivity used for network construction (default: correlation, options: correlation, partial correlation, tangent)')
parser.add_argument('--max_degree', type=int, default=8, help='Maximum Chebyshev polynomial degree.')
parser.add_argument('--ngl', default=8, type=int, help='number of gcn hidden layders (default: 8)')
parser.add_argument('--edropout', type=float, default=0.2, help='edge dropout rate')
parser.add_argument('--train', default=1, type=int, help='train(default: 1) or evaluate(0)')
parser.add_argument('--ckpt_path', type=str, default="c:\\Users\\HP\\Downloads\\abide\\ABIDE\\Folds", help='checkpoint path to save trained models')
parser.add_argument('--early_stopping', action='store_true', default=True, help='early stopping switch')
parser.add_argument('--early_stopping_patience', type=int, default=30, help='early stoppng epochs')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#After parsing arguments, you can add wandb.config to log the hyperparameters
device=torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
print("using device:",device)

np.random.seed(args.seed)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
params = dict()
params['no_cuda'] = args.no_cuda
params['seed'] = args.seed
params['epochs'] = args.epochs
params['lr'] = args.lr
params['weight_decay'] = args.weight_decay
params['hidden'] = args.hidden
params['dropout'] = args.dropout
params['atlas'] = args.atlas
params['num_features'] = args.num_features
params['folds'] = args.folds
params['connectivity'] = args.connectivity
params['max_degree'] = args.max_degree
params['ngl'] = args.ngl
params['edropout'] = args.edropout
params['train'] = args.train
params['ckpt_path'] = args.ckpt_path
params['early_stopping'] = args.early_stopping
params['early_stopping_patience'] = args.early_stopping_patience

#After parsing arguments, you can add wandb.config to log the hyperparameters

# Print Hyperparameters
print('Hyperparameters:')
for key, value in params.items():
    print(key + ":", value)

corrects = np.zeros(args.folds, dtype=np.int32) 
accs = np.zeros(args.folds, dtype=np.float32) 
aucs = np.zeros(args.folds, dtype=np.float32)
prfs = np.zeros([args.folds,3], dtype=np.float32) # Save Precision, Recall, F1
test_num = np.zeros(args.folds, dtype=np.float32)

print('  Loading dataset ...')
dataloader = dataloader()
raw_features, y, nonimg = dataloader.load_data(params) 
cv_splits = dataloader.data_split(args.folds)

t1 = time.time()

all_fold_accuracies = []
all_fold_aucs = []
all_fold_precisions = []
all_fold_recalls = []
all_fold_f1s = []
all_folds_roc_data = []

for i in range(args.folds):
    t_start = time.time()
    train_ind, test_ind = cv_splits[i]

    train_ind, valid_ind = train_test_split(train_ind, test_size=0.1, random_state = 24)
    
    cv_splits[i] = (train_ind, valid_ind)
    cv_splits[i] = cv_splits[i] + (test_ind,)
    print('Size of the {}-fold Training, Validation, and Test Sets:{},{},{}' .format(i+1, len(cv_splits[i][0]), len(cv_splits[i][1]), len(cv_splits[i][2])))


    all_labels=[]
    all_preds=[]
    included_samples=set()
    fold_accuracies1=[]
    fold_aucs1=[]
    fold_precisions1=[]
    fold_recall1=[]
    fold_f1s1=[]
    roc_data1=[]
     
    if args.train == 1: 
        for j in range(args.folds):
            print(' Starting the {}-{} Fold:：'.format(i+1,j+1))
            node_ftr = dataloader.get_node_features(train_ind)
            #H=construct_hypergraph(raw_features) #no.of nodes total 931 features->nodes
            edge_index, edgenet_input = dataloader.get_WL_inputs(nonimg,cv_splits,fold=0) # edge_index-> edges,edgenet_input -> feature for each edges
            edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
            features=(node_ftr-node_ftr.mean(axis=0))/node_ftr.std(axis=0)
            num_nodes = features.shape[0]  # Number of nodes
            graph = nx.Graph()
            graph.add_nodes_from(range(num_nodes))

            deepwalk = deepWalk(graph, walk_length=10, num_walks=80, embedding_dim=64)
            random_walk_embeddings=deepwalk.get_embeddings()
            print("Features shape:", features.shape)
            print("Edge index shape:", edge_index.shape)
            print("Edge net input shape:", edgenet_input.shape)
            print("Random walk embeddings shape:", random_walk_embeddings.shape)
            """
            model = GCN(input_dim = args.num_features,
                        nhid = args.hidden, 
                        num_classes = 2, 
                        ngl = args.ngl, 
                        dropout = args.dropout, 
                        edge_dropout = args.edropout, 
                        edgenet_input_dim = 2*nonimg.shape[1])
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            """

            model=HybridModel(
                input_dim=args.num_features,
                nhid=args.hidden,
                num_classes=2,
                ngl=args.ngl,
                dropout=args.dropout,
                edge_dropout=args.edropout,
                edgenet_input_dim=2*nonimg.shape[1],
                random_walk_dim=64
            )

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
           

            if args.cuda:
                model.to(device)
                #features=features.cuda()
                features = torch.tensor(node_ftr, dtype=torch.float32).to(device)
                edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
                edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(device)
                labels = torch.tensor(y, dtype=torch.long).to(device)
                random_walk_embeddings=torch.tensor(random_walk_embeddings,dtype=torch.float32).to(device)
                fold_model_path = args.ckpt_path + "/fold{}.pth".format(i+1)
                
            acc = 0
            best_val_loss = float('inf') # early stoppping: Initialized to positive infinity
            current_patience = 0 # early stopping: Used to record the epochs of the current early stopping

           # device=torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
            for epoch in range(args.epochs):
                # train
                node_ftr=dataloader.get_node_features(train_ind)
                features=torch.tensor(node_ftr,dtype=torch.float32).to(device)
                labels=torch.tensor(y,dtype=torch.long).to(device)
                
                model.to(device)
                model.train()
                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                    output, edge_weights = model(features, edge_index, edgenet_input,random_walk_embeddings)
                    loss_train = torch.nn.CrossEntropyLoss()(output[train_ind], labels[train_ind])
                    loss_train.backward()
                    optimizer.step()
                acc_train = torchmetrics_accuracy(output[train_ind], labels[train_ind])
                auc_train = torchmetrics_auc(output[train_ind], labels[train_ind])
                logits_train = output[train_ind].detach().cpu().numpy()
                prf_train = prf(logits_train, y[train_ind])


                # valid loop
                model.eval()
                with torch.set_grad_enabled(False):
                    output, edge_weights = model(features, edge_index, edgenet_input,random_walk_embeddings)
                loss_val = torch.nn.CrossEntropyLoss()(output[valid_ind], labels[valid_ind])
                acc_val = torchmetrics_accuracy(output[valid_ind], labels[valid_ind])
                auc_val = torchmetrics_auc(output[valid_ind], labels[valid_ind])
                logits_val = output[valid_ind].detach().cpu().numpy()
                preds_val=np.argmax(logits_val,axis=1)
                labels_val=labels[valid_ind].detach().cpu().numpy()
                preds_probs=output[valid_ind][:,1].detach().cpu().numpy()
                prf_val = prf(logits_val, y[valid_ind])

                all_folds_roc_data=[]


                fpr,tpr,thresholds=roc_curve(labels_val,preds_probs)
                roc_auc=auc(fpr,tpr)

                #if epoch not in roc_data:
                 #   roc_data[epoch]={}
                #roc_data[epoch][f"fold_{i+1}"]={"fpr":fpr,"tpr":tpr,"roc_auc":roc_auc}

                roc_data1.append({
                    "fpr":fpr,
                    "tpr":tpr,
                    "roc_auc":roc_auc,
                    "fold":i+1,
                    "epoch":epoch+1
                })


                # After each fold, collect metrics for box plots
    

                #all_preds.extend(preds_val)
                #all_labels.extend(labels_val)

                #for idx,pred,label in zip(valid_ind,preds_val,labels_val):
                 #   if idx not in included_samples:
                  #      all_preds.append(pred)
                   #     all_labels.append(label)
                    #    included_samples.add(idx)
                        
                                       # cm_val=confusion_matrix(labels_val,preds_val)
                #print(cm_val)

               
                
                print('Epoch:{:04d}'.format(epoch+1))
                print(f"Train Loss: {loss_train:.4f} | Train ACC : {acc_train :.4f} | Train AUC :{auc_train:.4f}")
                print(f"Val Loss :{loss_val:.4f} | Val ACC : {acc_val:.4f} ")
                print("-"*80)
                print('acc_train:{:.4f}'.format(acc_train),
                      'pre_train:{:.4f}'.format(prf_train[0]),
                      'recall_train:{:.4f}'.format(prf_train[1]),
                      'F1_train:{:.4f}'.format(prf_train[2]),
                      'AUC_train:{:.4f}'.format(auc_train))
                print('acc_val:{:.4f}'.format(acc_val),
                      'pre_val:{:.4f}'.format(prf_val[0]),
                      'recall_val:{:.4f}'.format(prf_val[1]),
                      'F1_val:{:4f}'.format(prf_val[2]),
                      'AUC_val:{:.4f}'.format(auc_val))
                



                
                if acc_val > acc :
                    acc = acc_val
                    ckpt_path="c:\\Users\\HP\\Downloads\\abide\\ABIDE\\Folds"
                    if not os.path.exists(ckpt_path):
                        os.makedirs(ckpt_path)

                    fold_model_path=os.path.join(ckpt_path,f"fold{j+1}.pth")

                    torch.save(model.state_dict(),fold_model_path)
                    print(f"model saved at {fold_model_path}")

                   # if args.ckpt_path != '':
                    #    if not os.path.exists(args.ckpt_path):
                     #       os.makedirs(args.ckpt_path)
                      #  torch.save(model.state_dict(), fold_model_path)
                
                # Early Stopping
                if epoch > 50 and args.early_stopping == True:
                    if loss_val < best_val_loss:
                        best_val_loss = loss_val
                        current_patience = 0
                    else:
                        current_patience += 1
                    if current_patience >= args.early_stopping_patience:
                        print('Early Stopping!!! epoch：{}'.format(epoch))
                        break


        # test
        print("Loading the Model for the {}-th Fold:... ...".format(i+1),
              "Size of samples in the test set:{}".format(len(test_ind)))
        if not os.path.exists(args.ckpt_path):
            os.makedirs(args.ckpt_path)
        
        fold_model_path=os.path.join(args.ckpt_path,"fold{}.pth".format(i+1))
        model.load_state_dict(torch.load(fold_model_path))
        model.eval()
        
        with torch.set_grad_enabled(False):
            output, edge_weights = model(features, edge_index, edgenet_input,random_walk_embeddings)
        acc_test = torchmetrics_accuracy(output[test_ind], labels[test_ind])
        auc_test = torchmetrics_auc(output[test_ind], labels[test_ind])
        logits_test = output[test_ind].detach().cpu().numpy()
        preds_test  = np.argmax(logits_test,axis=1)
        labels_test = labels[test_ind].detach().cpu().numpy()

        correct_test = correct_num(logits_test, y[test_ind])
        prf_test =  prf(logits_test, y[test_ind])

        all_preds.extend(preds_test)
        all_labels.extend(labels_test)

        acc_test=torchmetrics_accuracy(output[test_ind],labels[test_ind])
        auc_test=torchmetrics_accuracy(output[test_ind],labels[test_ind])
        prf_test=prf(logits_test,y[test_ind])
        fold_accuracies1.append(acc_test)
        fold_aucs1.append(auc_test)
        fold_precisions1.append(prf_test[0])
        fold_recall1.append(prf_test[1])
        fold_f1s1.append(prf_test[2])


                
        #plt.figure(figsize=(12,8))
        #sns.boxplot(data=metrics_df)
        #plt.title("Distribution of Metrics Across FOlds")
        #plt.ylabel("score")
        #plt.show()
        #wandb.log({"Box Plots":wandb.plot.box(metrics_df,title="Distribution of metrics across folds")})

        # log test metrices to wandb
        

        print("Test Accuracy: {:.4f},Test AUC: {:.4f}".format(acc_test,auc_test))
        print("Test Precision,Recall , F1: ",prf_test)
        
        t_end = time.time()
        t = t_end - t_start
        print('Fold {} Results:'.format(i+1),
              'test acc:{:.4f}'.format(acc_test),
              'test_pre:{:.4f}'.format(prf_test[0]),
              'test_recall:{:.4f}'.format(prf_test[1]),
              'test_F1:{:.4f}'.format(prf_test[2]),
              'test_AUC:{:.4f}'.format(auc_test),
              'time:{:.3f}s'.format(t))
        

        

        correct = correct_test
        aucs[i] = auc_test
        prfs[i] = prf_test
        corrects[i] = correct
        test_num[i] = len(test_ind)
    
    final_cm=confusion_matrix(labels_test,preds_test)

    print(len(preds_test))
    print(f"Confusion matric for test set.")
    #plt.show()
    print(final_cm)
    print(len(labels_test))

 
    
    plot_confusion_matrix(final_cm,title="confusion matrix for test set!")
    plt.show()

    plt.clf()
    
    all_fold_accuracies.extend(fold_accuracies1)
    all_fold_aucs.extend(fold_aucs1)
    all_fold_precisions.extend(fold_precisions1)
    all_fold_recalls.extend(fold_recall1)
    all_fold_f1s.extend(fold_f1s1)
    all_folds_roc_data.extend(roc_data1)

plt.figure(figsize=(10,8))
for fold_data in roc_data1:
    if fold_data["epoch"]==epoch+1:
        plt.plot(fold_data["fpr"],fold_data["tpr"],label=f'fold {fold_data["fold"]} (AUC={fold_data["roc_auc"]:.2f})')
    #plt.plot([0,1],[0,1],"k--")
plt.plot([0,1],[0,1],'k--')

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("roc curves for all folds")
plt.legend(loc="lower right")
plt.grid(True)
                

plt.show()
plt.clf()




# Log the box plot to wandb


 

t2 = time.time()

final_model_path = os.path.join(args.ckpt_path, "c:\\Users\\HP\\Downloads\\abide\\ABIDE\\final_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved at {final_model_path}")

checkpoint = {
    'epoch': args.epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss_train,  # Save the last training loss
    'accuracy': acc_train,  # Save the last training accuracy
    'auc': auc_train,  # Save the last training AUC
}
final_checkpoint_path = os.path.join(args.ckpt_path, "c:\\Users\\HP\\Downloads\\abide\\ABIDE\\final_model.pth")
torch.save(checkpoint, final_checkpoint_path)
print(f"Final checkpoint saved at {final_checkpoint_path}")

print('\r\n======Finish Results for Nested 10-fold cross-validation======')
Nested10kCV_acc = np.sum(corrects) / np.sum(test_num)
Nested10kCV_auc = np.mean(aucs)
Nested10kCV_precision, Nested10kCV_recall, Nested10kCV_F1 = np.mean(prfs, axis=0)
print('Test:',
      'acc:{}'.format(Nested10kCV_acc),
      'precision:{}'.format(Nested10kCV_precision),
      'recall:{}'.format(Nested10kCV_recall),
      'F1:{}'.format(Nested10kCV_F1),
      'AUC:{}'.format(Nested10kCV_auc))
print('Total duration:{}'.format(t2 - t1))
