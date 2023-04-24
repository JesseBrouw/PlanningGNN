import torch
import CustomDataset
import time
import ModulesGNN
import os
import numpy as np
import random
import sys
from sklearn.metrics import r2_score

from torch_geometric.loader import DataLoader

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

def main(model_type:str='dual'):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load graph_dataset
    dataset = CustomDataset.GraphDataset(root='labeled_data/', model=model_type)

    # split dataset
    train, val, test = dataset.alternative_split([0.7, 0.15, 0.15])

    begin = time.time()

    loss_function = torch.nn.MSELoss()
    # PARAMETERS                                         
    LEARNING_RATE = 0.005                 # learning rate of the optimizer
    EPOCHS = 30                           # number of epochs over the data
    BATCH_SIZE = 1

    # instantiate model
    model = ModulesGNN.GNN(
        model = model_type,
        lifted_num_layers = 3,
        lifted_graph_embedding_size = 32,
        lifted_operation = 'GAT',
        lifted_pool = 'max',
        grounded_num_layers = 3,
        grounded_graph_embedding_size = 32,
        grounded_operation = 'GAT',
        grounded_pool = 'max',
        hidden_dimension = 128,
        dropout = 0.5
    )
    # instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader =  DataLoader(dataset=val, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=True)

    losses_per_epoch = []
    for EPOCH in range(EPOCHS):
        model.train()
        losses = []
        for _, (lifted_batch, grounded_batch, domain) in enumerate(train_loader):
            LX = None
            GX = None 
            if model_type == 'lifted':
                LX = lifted_batch.to(device)
                y = lifted_batch.y.to(device)
            elif model_type == 'grounded':
                GX = grounded_batch.to(device)
                y = grounded_batch.y.to(device)
            else:
                LX = lifted_batch.to(device)
                GX = grounded_batch.to(device)
                y = lifted_batch.y.to(device)

            # compute output
            out = model(lifted_data=LX, grounded_data=GX)

            # compute loss
            loss = loss_function(out.to(torch.float), y.to(torch.float))
            losses.append(loss.item())

            # perform backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        E_loss = sum(losses)/len(losses)
        losses_per_epoch.append(E_loss)
        print(f'Train loss at epoch {EPOCH+1} = {E_loss}')
    
        val_losses = []
        predictions = []
        true = []
        for _, (lifted_batch, grounded_batch, domain) in enumerate(val_loader):
            with torch.no_grad():
                LX = None
                GX = None 
                if model_type == 'lifted':
                    LX = lifted_batch.to(device)
                    y = lifted_batch.y.to(device)
                elif model_type == 'grounded':
                    GX = grounded_batch.to(device)
                    y = grounded_batch.y.to(device)
                else:
                    LX = lifted_batch.to(device)
                    GX = grounded_batch.to(device)
                    y = lifted_batch.y.to(device)

                # compute output
                out = model(lifted_data=LX, grounded_data=GX)

                # compute loss
                loss = loss_function(out.to(torch.float), y.to(torch.float))
                val_losses.append(loss.item())

                predictions.append(out.to(torch.float).item())    # use of .item() necessary to avoid warnings and future errors
                true.append(y.to(torch.float).item())
    

    predictions = []
    true = []
    losses = []
    
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        for _, (lifted_batch, grounded_batch, domain) in enumerate(test_loader):
            LX = None
            GX = None 
            if model_type == 'lifted':
                LX = lifted_batch.to(device)
                y = lifted_batch.y.to(device)
            elif model_type == 'grounded':
                GX = grounded_batch.to(device)
                y = grounded_batch.y.to(device)
            else:
                LX = lifted_batch.to(device)
                GX = grounded_batch.to(device)
                y = lifted_batch.y.to(device)

            # compute output
            out = model(lifted_data=LX, grounded_data=GX)

            # compute loss
            loss = loss_fn(out.to(torch.float), y.to(torch.float))
            losses.append(loss.item())
            
            true.append(y.to(torch.float).item())
            predictions.append(out.to(torch.float).item())

    print(f'Mean loss on test set : {sum(losses)/len(losses)}')
    print(f'r2 score on the test set : {r2_score(true, predictions)}')



if __name__ == '__main__':
    model_type = sys.argv[1]
    if model_type not in ['dual', 'lifted', 'grounded']:
        print('First argument must be one of "dual", "lifted", "grounded"!')
    else:
        main(model_type)