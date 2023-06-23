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


"""
    Why transfer learning possibly does not work:
    - The convolution module appears to be responsible for the learning
    of the important features/information/underlying structures as it 
    achieves 91 percent of variance on the entire set. When I would 
    fix all the parameters present in the convolution module in the 
    transfer learning cycle on the target domain, the model is not 
    able to learn from the target domain using these important parameters, 
    while the domains are assumably quite different from each other. 
    
    Good experiment to try, but also quite computationally expensive. 
    Per domain also achieves quite good results in general.

    Maybe instead of fixing the parameters before transfer learning epochs, 
    use the parameters resulting from the initial cycle as starting point 
    for the transfer cycle : Deep domain adaptation.
"""

def main(model_type:str='dual', domain='driverlog'):
    # set device

    # grounded, batch_size=1 --> per EPOCH 40 seconds GPU, 116 seconds CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load graph_dataset
    dataset = CustomDataset.GraphDataset(root='labeled_data/', model=model_type)

    # split dataset
    train, transfer_train, transfer_test = dataset.transfer_split(domain=domain)

    begin = time.time()

    loss_function = torch.nn.MSELoss()
    # PARAMETERS                                         
    LEARNING_RATE = 0.0005                 # learning rate of the optimizer
    EPOCHS = 11                           # number of epochs over the data
    BATCH_SIZE = 1

    # instantiate model
    model2 = ModulesGNN.GNN(
        model = model_type,
        lifted_num_layers = 3,
        lifted_graph_embedding_size = 16,
        lifted_operation = 'GAT',
        lifted_pool = 'max',
        grounded_num_layers = 8,
        grounded_graph_embedding_size = 64,
        grounded_operation = 'GCN',
        grounded_pool = 'max',
        hidden_dimension = 1024,
        dropout = 0.1,
        device=device
    ).to(device)

    model = ModulesGNN.convolution_predictor(
        model = model_type,
        lifted_num_layers = 3,
        lifted_graph_embedding_size = 16,
        lifted_operation = 'GAT',
        lifted_pool = 'max',
        grounded_num_layers = 7,
        grounded_graph_embedding_size = 64,
        grounded_operation = 'GCN',
        grounded_pool = 'max',
        hidden_dimension = 1024,
        dropout = 0.2,
        device=device
    ).to(device)

    model2 = ModulesGNN.handcrafted_predictor(hidden_size=512)

    # instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
    transfer_train_loader =  DataLoader(dataset=transfer_train, batch_size=BATCH_SIZE, shuffle=True)
    transfer_test_loader = DataLoader(dataset=transfer_test, batch_size=BATCH_SIZE, shuffle=True)

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

            # perform backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        E_loss = sum(losses)/len(losses)
        losses_per_epoch.append(E_loss)
        print(f'Train loss at epoch {EPOCH+1} = {E_loss}')
    
    # preparations for transfer learning

    # # fix parameters
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # hidden = torch.nn.Linear(in_features=256, out_features=16)
    # final_layer = torch.nn.Linear(in_features=16, out_features=1)
    # model.predictor = torch.nn.Sequential(*list(model.predictor.children())[:-1], hidden, final_layer)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    losses_per_epoch = []

    # 4 epochs is determined empirically 
    for EPOCH in range(4):
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
        print(f'Transfer learning train loss at epoch {EPOCH+1} = {E_loss}')

    predictions = []
    true = []
    losses = []
    
    results_frame = pd.DataFrame(columns=['domain', 'file', 'true', 'prediction'])
    i = 0
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
            results_frame.loc[i] = [domain, file, t, p]
            i += 1

    print(f'Mean loss on test set : {sum(losses)/len(losses)}')
    print(f'r2 score on the test set : {r2_score(true, predictions)}')

    results_frame.to_csv(f'./results_transfer_learning/conv_grounded_results_{model_type}_{domain[0]}.csv', index=False)

    return (true, predictions)



if __name__ == '__main__':
    domains = ['barman', 'blocksworld', 'childsnack', 'data_network', 'depots', 'driverlog', 'floortile', 'gripper', 'hiking', 'logistics', 'maintenance', 'miconic', 'parking', 'rovers', 'scanalyzer', 'transport', 'visitall', 'woodworking', 'zenotravel']
    model_type = sys.argv[1]
    true = []
    predictions = []
    dom = []
    if model_type not in ['dual', 'lifted', 'grounded']:
        print('First argument must be one of "dual", "lifted", "grounded"!')
    else:
        for domain in domains:
            t, p = main(model_type, domain)
            true.extend(t)
            p.extend(p)
            dom.extend([domain for _ in t])
        results = pd.DataFrame(columns=['domain', 'true', 'predictions'])
        results['domain'] = dom
        results['true'] = t
        results['predictions'] = p
        results.to_csv('./results_transfer_learning/conv_grounded_results_4epochs.csv', index=False)

