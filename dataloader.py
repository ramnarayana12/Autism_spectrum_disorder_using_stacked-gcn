import data as Reader
import numpy as np
import torch
from utils import preprocess_features
from sklearn.model_selection import StratifiedKFold
from nilearn.connectome import ConnectivityMeasure
import os
from cnninte import CNNFeatureExtraction
from torch.utils.data import DataLoader,TensorDataset
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self,input_dim,latent_dim):
        super(AutoEncoder,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(input_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,latent_dim)
        )

        self.decoder=nn.Sequential(

         nn.Linear(latent_dim,512),
         nn.ReLU(),
         nn.Linear(512,1024),
         nn.ReLU(),
         nn.Linear(1024,input_dim)   
        )

    def forward(self,x):
        latent=self.encoder(x)
        reconstructed=self.decoder(latent)
        return latent,reconstructed
    


class dataloader():
    def __init__(self): 
        self.pd_dict = {}
        self.node_ftr_dim = 2000

        
        self.num_classes = 2 
        self.cnn_model=CNNFeatureExtraction(input_dim=111,output_dim=2000)

    def load_data(self, params, connectivity='correlation', atlas='ho'):
        ''' load multimodal data from ABIDE
        return: imaging features (raw), labels, non-image data
        '''

        subject_IDs = Reader.get_ids()
        labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
        num_nodes = len(subject_IDs)

        sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
        unique = np.unique(list(sites.values())).tolist()
        ages = Reader.get_subject_score(subject_IDs, score='AGE_AT_SCAN')
        genders = Reader.get_subject_score(subject_IDs, score='SEX') 

        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])
        site = np.zeros([num_nodes], dtype=int)
        age = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=int)

        for i in range(num_nodes):
            y_onehot[i, int(labels[subject_IDs[i]])-1] = 1
            y[i] = int(labels[subject_IDs[i]])
            site[i] = unique.index(sites[subject_IDs[i]])
            age[i] = float(ages[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]]
        
        self.y = y -1  

        self.raw_features = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)


#construct non-imaging feature matrix
        phonetic_data = np.zeros([num_nodes, 3], dtype=np.float32)
        phonetic_data[:,0] = site 
        phonetic_data[:,1] = gender 
        phonetic_data[:,2] = age 

# store into dict form to easy access

        self.pd_dict['SITE_ID'] = np.copy(phonetic_data[:,0])
        self.pd_dict['SEX'] = np.copy(phonetic_data[:,1])
        self.pd_dict['AGE_AT_SCAN'] = np.copy(phonetic_data[:,2]) 
        
        # returns fc matrix,labels,demogrpahic information
        return self.raw_features, self.y, phonetic_data

    def data_split(self, n_folds):
        # split data by k-fold CV
        skf = StratifiedKFold(n_splits=n_folds)
        cv_splits = list(skf.split(self.raw_features, self.y))
       # train_ind=cv_splits[folds][0]
       
        return cv_splits 
    """

    def train_autoencoder(self,features,latent_dim=2000,epochs=5,batch_size=10,learning_rate=1e-2):

        autoencoder=AutoEncoder(input_dim=features.shape[1],latent_dim=latent_dim)
        optimizer=torch.optim.Adam(autoencoder.parameters(),lr=learning_rate)
        criterion=nn.MSELoss()

        dataset=TensorDataset(torch.tensor(features,dtype=torch.float32))
        dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

        autoencoder.train()
        for epoch in range(epochs):
            epoch_loss =0
            for batch in dataloader:
                batch_features=batch[0]
                _,reconstructed=autoencoder(batch_features)
                loss=criterion(reconstructed,batch_features)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss+=loss.item()
            print(f"epoch {epoch+1}/{epochs},loss:{epoch_loss/len(dataloader)}")

        autoencoder.eval()
        with torch.no_grad():
            latent_features,_=autoencoder(torch.tensor(features,dtype=torch.float32))
        return latent_features.numpy()
    """

    def get_node_features(self, train_ind):
        '''preprocess node features for wl-deepgcn
        '''

        selected_features=self.raw_features[train_ind]
        print(f"selected features:",selected_features.shape)
        
        node_ftr = Reader.feature_selection(self.raw_features, self.y, train_ind, self.node_ftr_dim)
        print(f"Node feature shape after selection: {node_ftr.shape}")
#
        self.node_ftr = preprocess_features(node_ftr) 
        return self.node_ftr

    """
    def get_node_features(self,train_ind):

        train_ind=np.arange(927)
        selected_features=self.raw_features[train_ind]

        print(f"training features shape:{selected_features.shape}")

        latent_features=self.train_autoencoder(
            features=selected_features,
            latent_dim=self.node_ftr_dim,
            epochs=15,
            batch_size=15,
            learning_rate=1e-2
        )

        print(f"latent features shape with autoencoder:{latent_features.shape}")
        self.node_ftr=preprocess_features(latent_features)
        print(f"node features shape after autoencoder:{self.node_ftr.shape}")
        return self.node_ftr
        """
    def get_WL_inputs(self, nonimg,cv_splits,fold):
        '''get WL inputs for wl-deepgcn 
        '''
        # construct edge network inputs 
        n = self.node_ftr.shape[0] 
        num_edge = n*(1+n)//2 - n  # n*(n-1)//2,HO=6105
        pd_ftr_dim = nonimg.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64) 
        edgenet_input = np.zeros([num_edge, 2*pd_ftr_dim], dtype=np.float32)  
        aff_score = np.zeros(num_edge, dtype=np.float32)
        
        # static affinity score used to pre-prune edges 
       # train_ind=cv_splits[fold][0]
        aff_adj = Reader.get_static_affinity_adj(self.node_ftr, self.pd_dict)  
        flatten_ind = 0 
        for i in range(n):
            for j in range(i+1, n):
                edge_index[:,flatten_ind] = [i,j]
                edgenet_input[flatten_ind]  = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j]  
                flatten_ind +=1

        assert flatten_ind == num_edge, "Error in computing edge input"
        
        keep_ind = np.where(aff_score > 1.1)[0]  
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input

    
