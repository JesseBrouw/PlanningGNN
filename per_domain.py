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

    # grounded, batch_size=1 --> per EPOCH 40 seconds GPU, 116 seconds CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load graph_dataset
    dataset = CustomDataset.GraphDataset(root='labeled_data/', model=model_type)

    domains = [
            'barman',
            'blocksworld',
            'childsnack',
            'data_network',
            'depots',
            'driverlog',
            'floortile',
            'gripper',
            'hiking',
            'logistics',
            'maintenance',
            'miconic',
            'parking',
            'rovers',
            'scanalyzer',
            'transport',
            'visitall',
            'woodworking',
            'zenotravel'
        ]
    results_frame = pd.DataFrame(columns=['domain', 'true', 'prediction'])
    i = 0
    for domain in domains:
        # split dataset
        train, transfer_train, transfer_test = dataset.transfer_split(domain=domain)
        print(domain)

        begin = time.time()

        loss_function = torch.nn.MSELoss()
        # PARAMETERS                                         
        LEARNING_RATE = 0.0005               # learning rate of the optimizer
        EPOCHS = 10                           # number of epochs over the data
        BATCH_SIZE = 1

        # instantiate model
        model = ModulesGNN.convolution_predictor(
        model = model_type,
        lifted_num_layers = 9,
        lifted_graph_embedding_size = 16,
        lifted_operation = 'GAT',
        lifted_pool = 'max',
        grounded_num_layers = 10,
        grounded_graph_embedding_size = 128,
        grounded_operation = 'GCN',
        grounded_pool = 'max',
        hidden_dimension = 256,
        dropout = 0.1,
        device=device
        ).to(device)

        # instantiate model
        model2 = ModulesGNN.convolution_predictor(
            model = model_type,
            lifted_num_layers = 3,
            lifted_graph_embedding_size = 16,
            lifted_operation = 'GAT',
            lifted_pool = 'max',
            grounded_num_layers = 9,
            grounded_graph_embedding_size = 16,
            grounded_operation = 'GCN',
            grounded_pool = 'max',
            hidden_dimension = 512,
            dropout = 0.2,
            device=device
        ).to(device)

        model3 = ModulesGNN.handcrafted_predictor(hidden_size=512)
        # instantiate optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
        transfer_train_loader =  DataLoader(dataset=transfer_train, batch_size=BATCH_SIZE, shuffle=True)
        transfer_test_loader = DataLoader(dataset=transfer_test, batch_size=BATCH_SIZE, shuffle=True)

        losses_per_epoch = []
        for EPOCH in range(EPOCHS):
            model.train()
            losses = []
            for _, (lifted_batch, grounded_batch, domain, file) in tqdm(enumerate(transfer_train_loader), total=len(transfer_train_loader)):
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

                # perform backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            E_loss = sum(losses)/len(losses)
            losses_per_epoch.append(E_loss)
            print(f'Train loss at epoch {EPOCH+1} = {E_loss}')

        predictions = []
        true = []
        losses = []
        
        loss_fn = torch.nn.MSELoss()
        with torch.no_grad():
            for _, (lifted_batch, grounded_batch, domain, file) in enumerate(transfer_test_loader):
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
                results_frame.loc[i] = [domain, t, p]
                i += 1

        print(f'Mean loss on test set : {sum(losses)/len(losses)}')
        print(f'r2 score on the test set : {r2_score(true, predictions)}')

    results_frame.to_csv(f'./predictions/conv_per_domain_{model_type}_graphSizeSplit.csv', index=False)



if __name__ == '__main__':
    model_type = sys.argv[1]
    if model_type not in ['dual', 'lifted', 'grounded']:
        print('First argument must be one of "dual", "lifted", "grounded"!')
    else:
        main(model_type)