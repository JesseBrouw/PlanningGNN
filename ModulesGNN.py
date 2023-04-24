import torch
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv, GraphConv, TransformerConv, global_mean_pool, global_max_pool
import torch.nn.functional as F
    
class ConvolutionModule(torch.nn.Module):
    """
        Graph convolution module that learns the graph 
        representation features of the planning problems.

        Can be used on both the grounded and lifted 
        graphs, and returns the final graph embeddings,
        and the handcrafted features that are the percentages
        of the different node labels present in the entire graph. 
    """
    def __init__(self,
                 input_size,
                 num_layers,
                 dropout,
                 graph_embedding_size,
                 operation,
                 pool
                 ):
        super().__init__()
        operation_mapping = {
            'GAT' : GATConv,
            'GCN' : GCNConv,
            'TRANS' : TransformerConv,
            'GNN' : GraphConv
        }
        conv = operation_mapping[operation]

        pool_mapping = {
            'mean' : global_mean_pool,
            'max' : global_max_pool,
            #'attention' : AttentionalAggregation(conv(graph_embedding_size, graph_embedding_size))
        }
        self.pool = pool_mapping[pool]
        
        self.dropout = dropout
        self.conv1 = conv(input_size, graph_embedding_size)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(conv(graph_embedding_size, graph_embedding_size))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x.to(torch.float), data.edge_index, data.batch
        handcrafted = x.sum(dim=0)/x.sum(dim=0).sum()
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        x = self.pool(x, batch) 
        return x, handcrafted

class MultiLayeredPredictor(torch.nn.Module):
    """
        Feedforward neural network that inputs the calculated
        final embeddings and handcrafted features, and outputs
        the predicted minimal solvable horizon bound. 
    """
    def __init__(self,
                input_dimension,
                hidden_dimension,
                dropout
                ):
        super().__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(input_dimension, hidden_dimension)
        self.lin2 = torch.nn.Linear(hidden_dimension, 1)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def forward(self, input):
        x = F.relu(self.lin1(input))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


class GNN(torch.nn.Module):
    """"
        Model that predicts the minimal solvable horizon bound of planning
        problems by learning the features of the graph representations of 
        these problems.

        Options: model == dual --> Both lifted and grounded graphs are used.
                 model == lifted --> Only lifted graphs are used.
                 model == grounded --> Only grounded graphs are used.
    
    """
    def __init__(self,
                 model='dual',
                 lifted_num_layers=3,
                 lifted_graph_embedding_size=32,
                 lifted_operation='GAT',
                 lifted_pool='max',
                 grounded_num_layers=3,
                 grounded_graph_embedding_size=32,
                 grounded_operation='GAT',
                 grounded_pool='max',
                 hidden_dimension=128,
                 dropout=0.5
                 ):
        super().__init__()

        self.model = model

        self.lifted_conv = ConvolutionModule(
            input_size=16,
            num_layers=lifted_num_layers,
            dropout=dropout,
            graph_embedding_size=lifted_graph_embedding_size,
            operation=lifted_operation,
            pool=lifted_pool
        )
        self.grounded_conv = ConvolutionModule(
            input_size=7,
            num_layers=grounded_num_layers,
            dropout=dropout,
            graph_embedding_size=grounded_graph_embedding_size,
            operation=grounded_operation,
            pool=grounded_pool
        )
        
        input_dim_mapping = {
            'dual' : lifted_graph_embedding_size+16 + grounded_graph_embedding_size+7,
            'lifted' : lifted_graph_embedding_size+16,
            'grounded' : grounded_graph_embedding_size+7
        }
        self.predictor = MultiLayeredPredictor(
            input_dimension=input_dim_mapping[self.model],
            hidden_dimension=hidden_dimension,
            dropout=dropout
        )

    def forward(self, lifted_data=None, grounded_data=None):
        if self.model == 'dual':
            lifted_features, lifted_handcrafted = self.lifted_conv(lifted_data)
            grounded_features, grounded_handcrafted = self.grounded_conv(grounded_data)

            lifted_concat = torch.concat((lifted_features[0], lifted_handcrafted))
            grounded_concat = torch.concat((grounded_features[0], grounded_handcrafted))

            out = self.predictor(torch.concat((lifted_concat, grounded_concat)))

            return out
        elif self.model == 'lifted':
            lifted_features, lifted_handcrafted = self.lifted_conv(lifted_data)
            lifted_concat = torch.concat((lifted_features[0], lifted_handcrafted))
            out = self.predictor(lifted_concat)
            return out
        else:
            grounded_features, grounded_handcrafted = self.grounded_conv(grounded_data)
            grounded_concat = torch.concat((grounded_features[0], grounded_handcrafted))
            out = self.predictor(grounded_concat)
            return out
            







