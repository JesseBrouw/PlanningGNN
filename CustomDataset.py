import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
import random

random.seed(1)
class GraphDataset(Dataset):
    def __init__(self, root, model, leave_out=[]):
        self.root = root
        self.model = model
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
            'openstacks',
            'parking',
            'rovers',
            'scanalyzer',
            'transport',
            'visitall',
            'woodworking',
            'zenotravel'
        ]
        idx = 0
        self.datapoints = pd.DataFrame(columns=['domain', 'file'])

        if self.model == 'lifted':
            for domain in domains:
                for file in os.listdir(os.path.join(self.root, 'labeled_lifted_data', domain)):
                    self.datapoints.loc[idx] = [domain, file]
                    idx += 1
        elif self.model == 'grounded':
            for domain in domains:
                for file in os.listdir(os.path.join(self.root, 'labeled_grounded_data', domain)):
                    self.datapoints.loc[idx] = [domain, file]
                    idx += 1
        else:
            for domain in domains:
                lifted_files = set(os.listdir(os.path.join(self.root, 'labeled_lifted_data', domain)))
                grounded_files = set(os.listdir(os.path.join(self.root, 'labeled_grounded_data', domain)))
                union = list(lifted_files.intersection(grounded_files))

                for file in union:
                    self.datapoints.loc[idx] = [domain, file]
                    idx += 1
                        
    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, index):
        if self.model == 'dual':
            domain = self.datapoints.loc[index].domain
            lifted_path = os.path.join(self.root, 'labeled_lifted_data', domain, self.datapoints.loc[index].file)
            grounded_path = os.path.join(self.root, 'labeled_grounded_data', domain, self.datapoints.loc[index].file)

            lifted_data = torch.load(lifted_path)
            grounded_data = torch.load(grounded_path)
            return lifted_data, grounded_data, domain
        elif self.model == 'lifted':
            domain = self.datapoints.loc[index].domain
            path = os.path.join(self.root, 'labeled_lifted_data', domain, self.datapoints.loc[index].file)
            lifted_data = torch.load(path)

            return lifted_data, lifted_data, domain
        else:
            domain = self.datapoints.loc[index].domain
            path = os.path.join(self.root, 'labeled_grounded_data', domain, self.datapoints.loc[index].file)
            grounded_data = torch.load(path)

            return grounded_data, grounded_data, domain

    def alternative_split(self, ratios:tuple, shuffle=False):
        train, val, test = ratios
        train_set = []
        val_set = []
        test_set = []
        domains = self.datapoints.domain.unique()

        for domain in domains:
            subset_indices = self.datapoints.index[self.datapoints.domain == domain].to_list()
            if shuffle:
                random.shuffle(subset_indices)
            domain_length = len(subset_indices)
            train_set.append(subset_indices[:int(train*domain_length)])
            val_set.append(subset_indices[int(train*domain_length):int(train*domain_length)+int(val*domain_length)])
            test_set.append(subset_indices[int(train*domain_length)+int(val*domain_length):])

        train_set = [item for items in train_set for item in items]
        test_set = [item for items in test_set for item in items]
        val_set = [item for items in val_set for item in items]

        return Subset(self, train_set), Subset(self, val_set), Subset(self, test_set)

    def transfer_split(self, domain='driverlog'):
        domains = self.datapoints.domain.unique()

        domains = [i for i in domains if i != domain]
        
        train_set = self.datapoints.index[self.datapoints.domain != domain].to_list()
        transfer_set = self.datapoints.index[self.datapoints.domain == domain].to_list()
        
        domain_length = len(transfer_set)
        transfer_train_set = transfer_set[:int(0.85*domain_length)]
        transfer_test_set = transfer_set[int(0.85*domain_length):]

        return Subset(self, train_set), Subset(self, transfer_train_set), Subset(self, transfer_test_set)




        




