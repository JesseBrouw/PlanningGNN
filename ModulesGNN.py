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
                 pool,
                 device
                 ):
        super().__init__()
        operation_mapping = {
            'GAT' : GATConv(input_size, graph_embedding_size),
            'GCN' : GCNConv(input_size, graph_embedding_size, improved=True),
            'TRANS' : TransformerConv(input_size, graph_embedding_size),
            'GNN' : GraphConv(input_size, graph_embedding_size)
        }

        operation_mapping2 = {
            'GAT' : GATConv(graph_embedding_size, graph_embedding_size),
            'GCN' : GCNConv(graph_embedding_size, graph_embedding_size, improved=True),
            'TRANS' : TransformerConv(graph_embedding_size, graph_embedding_size),
            'GNN' : GraphConv(graph_embedding_size, graph_embedding_size)
        }
        conv1 = operation_mapping[operation]
        convs = operation_mapping2[operation]

        pool_mapping = {
            'mean' : global_mean_pool,
            'max' : global_max_pool,
            #'attention' : AttentionalAggregation(conv(graph_embedding_size, graph_embedding_size))
        }
        self.pool = pool_mapping[pool]
        self.device = device
        self.dropout = dropout
        self.conv1 = conv1
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(convs)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x.to(torch.float), data.edge_index, data.batch
        graph_size = x.sum(dim=0).sum()
        handcrafted = torch.concat((x.sum(dim=0)/graph_size, torch.log(graph_size).unsqueeze(dim=0)))
        # handcrafted = (x.sum(dim=0)/graph_size)
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        x = self.pool(x, batch) 
        return x, handcrafted

class handcrafted_predictor(torch.nn.Module):
    """
        Module made to test the predictive power of only using the handcrafted features
        and leaving out the graph convolutions.
    """

    def __init__(self,
                 hidden_size
                 ):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=25, out_features=hidden_size)
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=128)
        self.fc3 = torch.nn.Linear(in_features=128, out_features=1)

    def forward(self, lifted_data=None, grounded_data=None):
        lifted_x, lifted_edge_index, lifted_bactch = lifted_data.x.to(torch.float), lifted_data.edge_index, lifted_data.batch
        grounded_x, grounded_edge_index, grounded_bactch = grounded_data.x.to(torch.float), grounded_data.edge_index, grounded_data.batch

        lifted_graph_size = lifted_x.sum(dim=0).sum()
        lifted_handcrafted = torch.concat((lifted_x.sum(dim=0)/lifted_graph_size, torch.log(lifted_graph_size).unsqueeze(dim=0)))
        #lifted_handcrafted = torch.concat((torch.log(lifted_x.sum(dim=0)), torch.log(lifted_graph_size).unsqueeze(dim=0)))
        grounded_graph_size = grounded_x.sum(dim=0).sum()
        grounded_handcrafted = torch.concat((grounded_x.sum(dim=0)/grounded_graph_size, torch.log(grounded_graph_size).unsqueeze(dim=0)))
        #grounded_handcrafted = torch.concat((torch.log(grounded_x.sum(dim=0)), torch.log(grounded_graph_size).unsqueeze(dim=0)))

        # print(grounded_handcrafted)

        out = self.fc1(torch.concat((lifted_handcrafted, grounded_handcrafted)))
        out = self.fc2(out)
        out = self.fc3(out)

        return out

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

class convolution_predictor(torch.nn.Module):
    """
        Does not use handcrafted features, only the convolution modules.
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
                 dropout=0.5,
                 device='cpu'
                 ):
        super().__init__()

        self.model = model

        self.lifted_conv = ConvolutionModule(
            input_size=16,
            num_layers=lifted_num_layers,
            dropout=dropout,
            graph_embedding_size=lifted_graph_embedding_size,
            operation=lifted_operation,
            pool=lifted_pool,
            device=device
        )
        self.grounded_conv = ConvolutionModule(
            input_size=7,
            num_layers=grounded_num_layers,
            dropout=dropout,
            graph_embedding_size=grounded_graph_embedding_size,
            operation=grounded_operation,
            pool=grounded_pool,
            device=device
        )
        
        input_dim_mapping = {
            'dual' : lifted_graph_embedding_size + grounded_graph_embedding_size,
            'lifted' : lifted_graph_embedding_size,
            'grounded' : grounded_graph_embedding_size
        }
        self.predictor = MultiLayeredPredictor(
            input_dimension=input_dim_mapping[self.model],
            hidden_dimension=hidden_dimension,
            dropout=dropout
        )

    def forward(self, lifted_data=None, grounded_data=None):
        if self.model == 'dual':
            lifted_features, _ = self.lifted_conv(lifted_data)
            grounded_features, _ = self.grounded_conv(grounded_data)
            out = self.predictor(torch.concat((lifted_features[0], grounded_features[0])))
            return out
        
        elif self.model == 'lifted':
            lifted_features, _ = self.lifted_conv(lifted_data)
            out = self.predictor(lifted_features)
            return out
        else:
            grounded_features, _ = self.grounded_conv(grounded_data)
            out = self.predictor(grounded_features)
            return out

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
                 dropout=0.5,
                 device='cpu'
                 ):
        super().__init__()

        self.model = model

        self.lifted_conv = ConvolutionModule(
            input_size=16,
            num_layers=lifted_num_layers,
            dropout=dropout,
            graph_embedding_size=lifted_graph_embedding_size,
            operation=lifted_operation,
            pool=lifted_pool,
            device=device
        )
        self.grounded_conv = ConvolutionModule(
            input_size=7,
            num_layers=grounded_num_layers,
            dropout=dropout,
            graph_embedding_size=grounded_graph_embedding_size,
            operation=grounded_operation,
            pool=grounded_pool,
            device=device
        )
        
        input_dim_mapping = {
            'dual' : lifted_graph_embedding_size+17 + grounded_graph_embedding_size+8,
            'lifted' : lifted_graph_embedding_size+17,
            'grounded' : grounded_graph_embedding_size+8
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
            







