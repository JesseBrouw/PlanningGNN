import torch
from tqdm import tqdm
import numpy as np
import os
import time
from sklearn.metrics import r2_score, mean_squared_error
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from datetime import date
import ModulesGNN
import sys
import random

import metrics as metrics
from CustomDataset import GraphDataset
from torch_geometric.loader import DataLoader

#TODO: determine way to sort on complexity; graph size? 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(1)

def load_data(data_root, model_type):
    # load the data
    dataset = GraphDataset(root=f'{data_root}/labeled_data/', model=model_type, leave_out=[])
    return dataset.alternative_split((0.7, 0.15, 0.15), shuffle=True)

def quadquad_loss(pred, y, p=2, alpha=0.5):
    """
        Estimation_of_flood_warning_runoff_thresholds_in_u.pdf

        Assymetric loss function, penalizes underestimating more when
        alpha > 0.5, less when alpha = 0.5, and for p = 2, and alpha = 0.5
        it is mean squared error. 
    """
    return 2*(alpha + (1-2*alpha)*((y-pred)<0)) * torch.abs((y-pred))**p

def main(operation, model_type, num_samples=15, max_num_epochs=20):
    # Save root where datafiles reside
    data_root = os.getcwd()

    # Create dictionary with the parameters that are tuned
    config = {
        "n_iter": tune.sample_from(lambda _: np.random.randint(1, 10)),
        "embedding_size": tune.sample_from(lambda _: 2 ** np.random.randint(4, 6)),
        "hidden_size": tune.sample_from(lambda _: 2 ** np.random.randint(6, 10)),
        "lr": tune.loguniform(1e-4, 1e-3),
        "dropout": tune.choice([0.1, 0.2, 0.4, 0.5]),
        "epochs": max_num_epochs,
        "operation": operation,
        "pool": tune.choice(['mean', 'max']),
        "model_type": model_type
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
        )
    
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"]
        )
    
    result = tune.run(
        partial(train, data_root=data_root),
        resources_per_trial={"cpu": 4},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
        )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation R^2 score: {}".format(
        best_trial.last_result["r2"]))

    best_trained_model = ModulesGNN.GNN(
        model=model_type,
        lifted_num_layers = best_trial.config["n_iter"], 
        lifted_graph_embedding_size=best_trial.config['embedding_size'],
        lifted_operation=best_trial.config['operation'],
        lifted_pool=best_trial.config['pool'],
        grounded_num_layers = best_trial.config["n_iter"], 
        grounded_graph_embedding_size=best_trial.config['embedding_size'],
        grounded_operation=best_trial.config['operation'],
        grounded_pool=best_trial.config['pool'],
        hidden_dimension=best_trial.config['hidden_size'],
        dropout = best_trial.config['dropout']
        )
    device = "cpu"
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.dir_or_data
    model_state, _ = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    _, _, test_set = load_data(data_root, model_type)

    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)

    test_r2, test_loss = test_accuracy(best_trained_model, test_loader, model_type)
    name = ''
    for key, value in best_trial.config.items():
        name += f'{key}:{value}_'

    # Save as under, mse or over, depending on the chosen value of alpha for the quadquad loss.
    torch.save(best_trained_model, os.path.join(os.getcwd(), 'saved_models', model_type, 'mse', f'{operation}_{date.today().strftime("%m-%d-%y")}_{test_r2}_{test_loss}_{name}.pt'))

    print(f'Best model parameters : \n {best_trained_model.parameters}')


def test_accuracy(model, loader, model_type, device="cpu"):
    predictions = []
    true = []
    losses = []
    
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        for _, (lifted_batch, grounded_batch, domain) in enumerate(loader):
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

    return r2_score(true, predictions), sum(losses)/len(losses)

def train(config, checkpoint_dir=None, data_root=None):
    loss_function = torch.nn.MSELoss()
    # PARAMETERS
    LEARNING_RATE = config['lr']           # learning rate of the optimizer
    EPOCHS = config['epochs']                            # number of epochs over the data
    BATCH_SIZE = 1

    model_type = config['model_type']

    # instantiate model
    model = ModulesGNN.GNN(
        model=model_type,
        lifted_num_layers = config["n_iter"], 
        lifted_graph_embedding_size=config['embedding_size'],
        lifted_operation=config['operation'],
        lifted_pool=config['pool'],
        grounded_num_layers = config["n_iter"], 
        grounded_graph_embedding_size=config['embedding_size'],
        grounded_operation=config['operation'],
        grounded_pool=config['pool'],
        hidden_dimension=config['hidden_size'],
        dropout = config['dropout']
        )
    # instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
            )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    train_set, val_set, test_set = load_data(data_root, model_type)

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader =  DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

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
        
        with tune.checkpoint_dir(EPOCH) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        
        
        tune.report(loss=(sum(val_losses)/len(val_losses)), r2=r2_score(true, predictions), mse=mean_squared_error(true, predictions))
    return test_loader


if __name__ == '__main__':
    operation = sys.argv[1]
    model_type = sys.argv[2]
    if model_type not in ['lifted', 'grounded', 'dual']:
        print('Second argument must be one of "lifted", "grounded" or "dual"!')
    else:
        main(operation, model_type)