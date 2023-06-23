import torch
import CustomDataset
import time
import ModulesGNN
import os
import numpy as np
import random
import sys
from sklearn.metrics import r2_score
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

from torch_geometric.loader import DataLoader


"""
    File which runs the model once with a given parameter configuration. 
    Saves results on the test set in ./predictions/Results.csv.

    Usage : python RunModel.py dual|lifted|grounded
"""


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

    # grounded, batch_size=1 --> per EPOCH 40 seconds GPU, 116 seconds CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load graph_dataset
    dataset = CustomDataset.GraphDataset(root='labeled_data/', model=model_type)

    # split dataset
    train, val, test = dataset.graph_size_split([0.7, 0.15, 0.15])

    begin = time.time()

    loss_function = torch.nn.MSELoss()
    # PARAMETERS                                         
    LEARNING_RATE = 0.0005                 # learning rate of the optimizer
    EPOCHS = 11                           # number of epochs over the data
    BATCH_SIZE = 1

    model2 = ModulesGNN.handcrafted_predictor(256)
    # instantiate model
    model2 = ModulesGNN.GNN(
        model = model_type,
        lifted_num_layers = 9,
        lifted_graph_embedding_size = 16,
        lifted_operation = 'GCN',
        lifted_pool = 'max',
        grounded_num_layers = 8,
        grounded_graph_embedding_size = 128,
        grounded_operation = 'GCN',
        grounded_pool = 'max',
        hidden_dimension = 256,
        dropout = 0.1,
        device=device
    ).to(device)

    model = ModulesGNN.convolution_predictor(
        model = model_type,
        lifted_num_layers = 9,
        lifted_graph_embedding_size = 16,
        lifted_operation = 'GAT',
        lifted_pool = 'max',
        grounded_num_layers = 7,
        grounded_graph_embedding_size = 64,
        grounded_operation = 'GCN',
        grounded_pool = 'max',
        hidden_dimension = 1024,
        dropout = 0.1,
        device=device
    ).to(device)
    # instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader =  DataLoader(dataset=val, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=True)
    
    train_true = defaultdict(list)
    losses_per_epoch = []
    for EPOCH in range(EPOCHS):
        model.train()
        losses = []
        for _, (lifted_batch, grounded_batch, domain, file) in tqdm(enumerate(train_loader), total=len(train_loader)):
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
            loss = loss_function(out.to(torch.float).squeeze(0), y.to(torch.float))
            losses.append(loss.item())

            train_true[domain].append(round(y.to(torch.float).item()))
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
        for _, (lifted_batch, grounded_batch, domain, file) in tqdm(enumerate(val_loader), total=len(val_loader)):
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
                loss = loss_function(out.to(torch.float).squeeze(0), y.to(torch.float))
                val_losses.append(loss.item())

                predictions.append(round(out.to(torch.float).item()))    # use of .item() necessary to avoid warnings and future errors
                true.append(round(y.to(torch.float).item()))
        print(f'Mean loss on validation set : {sum(val_losses)/len(val_losses)}')
        print(f'r2 score on the validation set : {r2_score(true, predictions)}')
    

    predictions = []
    true = []
    losses = []
    
    results_frame = pd.DataFrame(columns=['domain', 'true', 'prediction', 'benchmark_prediction'])
    i = 0
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        for _, (lifted_batch, grounded_batch, domain, file) in enumerate(test_loader):
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
            loss = loss_fn(out.to(torch.float).squeeze(0), y.to(torch.float))
            losses.append(loss.item())
            
            t = round(y.to(torch.float).item())
            p = round(out.to(torch.float).item())
            true.append(t)
            predictions.append(p)
            results_frame.loc[i] = [domain, t, p, round(np.mean(train_true[domain]))]
            i += 1

    print(f'Mean loss on test set : {sum(losses)/len(losses)}')
    print(f'r2 score on the test set : {r2_score(true, predictions)}')
    print(f'Train losses per epoch : {losses_per_epoch}')
    print(f'Validation losses per epoch : {val_losses}')

    results_frame.to_csv('./predictions/gcn_grounded_conv_graphSizeSplit.csv', index=False)



if __name__ == '__main__':
    model_type = sys.argv[1]
    if model_type not in ['dual', 'lifted', 'grounded']:
        print('First argument must be one of "dual", "lifted", "grounded"!')
    else:
        main(model_type)