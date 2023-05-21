import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import GCNConv ,GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torchvision.models as models
# torch.manual_seed(42)

class GNN(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(GNN, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        dense_neuron = model_params['model_dense_neurons']
        self.drop_outrate = model_params["model_dropout_rate"]
        self.n_layers = model_params['model_layers_size']
        self.avgPool = torch.nn.AdaptiveAvgPool2d((1,1))
        
        
        self.initial_layer = GCNConv(2,embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)
        self.conv_layers = ModuleList([])
        self.bn_layers = ModuleList([])
        # Layers
        for i in range(self.n_layers):
            self.conv_layers.append(GCNConv(embedding_size,embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size)) 

    
        # classifier
        self.fc1 = Linear(embedding_size*2, dense_neuron,bias = True)
        self.b1 = BatchNorm1d(dense_neuron)
        self.fc2 = Linear(dense_neuron, int(dense_neuron/2) )
        self.b2 = BatchNorm1d(int(dense_neuron/2))
        self.fc3 = Linear(int(dense_neuron/2), 7)  


    def forward(self, x, edge_attr, edge_index, batch_index):
        # edge_index, _ = remove_self_loops(edge_index)
        # # # # Holds the intermediate graph representations
        global_representation = []

        edge_attr = edge_attr.abs()
        # initial conv
        x = self.initial_layer(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.bn1(x)
        
        for i in range(self.n_layers):
            x = self.conv_layers[i](x,edge_index,edge_attr)
            x = torch.relu(x)
            x = self.bn_layers[i](x)
            global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))

    
    
        x = sum(global_representation )

        w = torch.relu(self.fc1(x))
        w = self.b1(w)
        w = F.dropout(w, p=self.drop_outrate, training=self.training)
        w = torch.relu(self.fc2(w))
        w = self.b2(w)
        w = F.dropout(w, p=self.drop_outrate, training=self.training)
        w = F.log_softmax(self.fc3(w), dim=1)

        return w