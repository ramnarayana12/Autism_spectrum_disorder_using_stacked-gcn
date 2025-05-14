import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear as Lin, Sequential as Seq
import torch_geometric as tg
from wl import WL
from torch_geometric.utils import add_self_loops

import  matplotlib.pyplot as plt
from torch_geometric.nn import ChebConv
device=torch.device("cuda" if torch.cuda.is_available()  else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim, nhid,output_dim):
        super(MLP,self).__init__()
        self.cls = nn.Sequential(
            torch.nn.Linear(input_dim,nhid),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(nhid),
            torch.nn.Linear(nhid,output_dim)
        )
        
    def forward(self, features):
        return self.cls(features)
    
class StackedGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(StackedGCN, self).__init__()
        self.gconv = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            self.gconv.append(ChebConv(in_channels, hidden_dim, K=8, normalization='sym'))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_index, edge_weight):

        print(f"input x shape:{x.shape}")
        print(f"edge index shape:{edge_index.shape}")
        print(f"edge weight shape:{edge_weight.shape}")


        if isinstance(x,np.ndarray):
            x=torch.tensor(x,dtype=torch.float,device=edge_index.device if torch.is_tensor(edge_index) else "cpu")
        
            print(f"converted x to tensor. new type :{type(x)}")
        if isinstance(edge_index,np.ndarray):
            edge_index=torch.tensorf(edge_index,dtype=torch.long,device=x.device)
            print(f"converted edge_index to tensor. new type :{type(edge_index)}")

        if x.dim() != 2:
            raise ValueError(f"Expected x to be 2D, got {x.dim()}D")
        if edge_index.shape[0] != 2:
            raise ValueError(f"Expected edge_index to have shape (2, num_edges), got {edge_index.size()}")
        if edge_weight.dim() !=1:
            raise ValueError(f"Expected edge weight to be 1D,got {edge_weight.dim()}D")
                
        print(f"Edge index (first 5): {edge_index[:, :5]}")
        print(f"Edge weight (first 5): {edge_weight[:5]}")

        
        for conv in self.gconv:

            print(f"before chebconv - x shape :{x.shape},edge_index shape:{edge_index.shape},edge_weight shape:{edge_weight.shape}")

            x = self.relu(conv(x, edge_index, edge_weight))
            print(f"after chebconv -x shape:{x.shape}")

        return x

class GCN(nn.Module):
    def __init__(self, input_dim, nhid, num_classes, ngl, dropout, edge_dropout, edgenet_input_dim):
        super(GCN, self).__init__()
        K=3   
        hidden = [nhid for i in range(ngl)] 
        self.dropout = dropout
        self.edge_dropout = edge_dropout 
        bias = False 
        self.relu = torch.nn.ReLU(inplace=True) 
        self.ngl = ngl 
        self.gconv = nn.ModuleList()
        for i in range(ngl):
            in_channels = input_dim if i==0  else hidden[i-1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K, normalization='sym', bias=bias)) 
          
        self.cls = nn.Sequential(
                torch.nn.Linear(nhid, 128),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(128), 
                torch.nn.Linear(128, num_classes))

        self.edge_net = WL(input_dim=edgenet_input_dim//2, dropout=dropout)
        #self.model_init()

        # retain information loss in resdiual connections at DEEPGCN


    def add_self_loops_with_weights(self,edge_index,edge_weight,num_nodes):
        loop_index=torch.arange(0,num_nodes,device=edge_index.device).unsqueeze(0).repeat(2,1)
        loop_weight=torch.ones(num_nodes,device=edge_weight.device)

        edge_index=torch.cat([edge_index,loop_index],dim=1)
        edge_weight=torch.cat([edge_weight,loop_weight],dim=0)

        return edge_index,edge_weight

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight) # He init
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
    
    def forward(self, features, edge_index, edgenet_input, enforce_edropout=False): 
        
        if isinstance(edge_index,np.ndarray):
            edge_index=torch.tensor(edge_index,dtype=torch.long,device=features.device)
        
        if isinstance(edgenet_input,np.ndarray):
            edgenet_input=torch.tensor(edgenet_input,dtype=torch.float,device=features.device)
        

        num_nodes=features.size(0)
        if self.edge_dropout>0:
            if enforce_edropout or self.training:
                one_mask = torch.ones([edgenet_input.shape[0],1]).to(device)
                self.drop_mask = F.dropout(one_mask, self.edge_dropout, True)
                self.bool_mask = torch.squeeze(self.drop_mask.type(torch.bool))
                edge_index = edge_index[:, self.bool_mask] 
                edgenet_input = edgenet_input[self.bool_mask] # Weights

               # print("drop mask shape:", self.drop_mask.shape)
               # print("bool mask shape:",self.bool_mask.shape)
                #print("edge index shape after dropout:",edge_index.shape)
                #print("edgenet input shape after dropout:",edgenet_input.shape)

            
        #edge_weight = torch.squeeze(self.edge_net(edgenet_input))
        edge_weight=self.edge_net(edgenet_input).view(-1)

        edge_index,edge_weight=self.add_self_loops_with_weights(edge_index,edge_weight,num_nodes)


       # print("edge index shape:", edge_index.shape)
       # print("edge weight shape:", edge_weight.shape)
        
        #num_nodes=features.size(0)


        # GCN residual connection
        # input layer
        features = F.dropout(features, self.dropout, self.training)
        x = self.relu(self.gconv[0](features, edge_index, edge_weight)) 
        x_temp = x
        
        # hidden layers
        for i in range(1, self.ngl - 1): # self.ngl→7
            x = F.dropout(x_temp, self.dropout, self.training)
            x = self.relu(self.gconv[i](x, edge_index, edge_weight)) 
            x_temp = x_temp + x # ([871,64])

        # output layer
        x = F.dropout(x_temp, self.dropout, self.training)
        x = self.relu(self.gconv[self.ngl - 1](x, edge_index, edge_weight))
        x_temp = x_temp + x

        output = x # Final output is not cumulative
        output = self.cls(output) 
        
        return output, edge_weight
    
    ## num_nodes = features.size(0)
       # edge_weight = self.edge_net(edgenet_input).view(-1)
       # edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)

        # GCN forward pass
       # x = features
       # for i in range(self.ngl):
          #  x = F.dropout(x, self.dropout, self.training)
          #  x = self.relu(self.gconv[i](x, edge_index, edge_weight))
      #  output = self.cls(x)
        #return output, edge_weight

class HybridModel(nn.Module):
    def __init__(self, input_dim, nhid, num_classes, ngl, dropout, edge_dropout, edgenet_input_dim, random_walk_dim):
        super(HybridModel, self).__init__()
        self.stacked_gcn = StackedGCN(random_walk_dim, nhid, ngl)
        self.deep_gcn = GCN(input_dim, nhid, num_classes, ngl, dropout, edge_dropout, edgenet_input_dim)
        #self.mlp = MLP(nhid * 2, 128, num_classes)  # Combine Stacked GCN and DeepGCN outputs
        self.mlp=MLP(nhid + num_classes,128,num_classes)
        self.dropout = dropout
    
    def forward(self, features, edge_index, edgenet_input, random_walk_embeddings):
        #if isinstance(random_walk_embeddings, np.ndarray):
          #  random_walk_embeddings = torch.tensor(random_walk_embeddings, dtype=torch.float, device=features.device)
          #  print(f"Converted random_walk_embeddings to tensor. New type: {type(random_walk_embeddings)}")
        

        if isinstance(edge_index, np.ndarray):
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=features.device)
            print(f"Converted edge_index to tensor. New type: {type(edge_index)}")

        edge_weight = self.deep_gcn.edge_net(edgenet_input).view(-1)

        # Stacked GCN forward pass (using random walk embeddings)
        stackedgcn_output = self.stacked_gcn(random_walk_embeddings, edge_index, edge_weight)

        # DeepGCN forward pass (using raw features)
        deepgcn_output, _ = self.deep_gcn(features, edge_index, edgenet_input)
        print("Stacked GCN output shape:", stackedgcn_output.shape)
        print("Deep GCN output shape:", deepgcn_output.shape)

        # Ensure both outputs have the same number of nodes
        assert stackedgcn_output.shape[0] == deepgcn_output.shape[0], "Mismatch in number of nodes!"

        # Concatenate outputs
        combined_output = torch.cat([stackedgcn_output, deepgcn_output], dim=1)
                # Initialize MLP dynamically
       
        # Final MLP for classification
        output = self.mlp(combined_output)
        return output, edge_weight
    
    """

    def forward(self, features, edge_index, edgenet_input, random_walk_embeddings):
    # Convert random_walk_embeddings to PyTorch tensor if it's a NumPy array
        if isinstance(random_walk_embeddings, np.ndarray):
            random_walk_embeddings = torch.tensor(random_walk_embeddings, dtype=torch.float, device=features.device)

        edge_weight = self.deep_gcn.edge_net(edgenet_input).view(-1)
        print("edge weight shape:",edge_weight.shape)

    # Stacked GCN forward pass (using random walk embeddings)
        stackedgcn_output = self.stacked_gcn(random_walk_embeddings, edge_index, edge_weight)

    # DeepGCN forward pass (using raw features)
        deepgcn_output, _ = self.deep_gcn(features, edge_index, edgenet_input)
        print("Stacked GCN output shape:", stackedgcn_output.shape)
        print("Deep GCN output shape:", deepgcn_output.shape)

    # Ensure both outputs have the same number of nodes
        assert stackedgcn_output.shape[0] == deepgcn_output.shape[0], "Mismatch in number of nodes!"

    # Concatenate outputs
        combined_output = torch.cat([stackedgcn_output, deepgcn_output], dim=1)
    
        # Final MLP for classification
        output = self.mlp(combined_output)
        return output, edge_weight
    """