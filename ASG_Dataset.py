from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch
import json
import os


class ASGDataset(Dataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        return 'graph_dataset.json'

    @property
    def processed_file_names(self):
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

        return [os.listdir(os.path.join(self.root, domain)) for domain in domains]

    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'r') as rf:
            self.dataset = json.load(rf)
        
        idx = 0
        for domain in self.dataset.keys():
            for problem in self.dataset[domain].keys():
                # create Data object

                x = torch.tensor(self.dataset[domain][problem]['x'], dtype=torch.float)
                edge_index = torch.tensor(self.dataset[domain][problem]['edge_index'], dtype=torch.long)
                label = torch.tensor(self.dataset[domain][problem]['horizon'])
                
                data = Data(x=x,
                            edge_index=edge_index,
                            y=label
                            )
                
                torch.save(data, 
                           os.path.join(self.processed_dir, f'data_{idx}.pt')
                           )
                idx += 1

                torch.save(data,
                           os.path.join(self.root, domain, f'{problem}.pt' )    
                )
                with open(os.path.join(self.root, 'datapoint_mappings.txt'), 'a') as wf:
                    wf.write(f'{domain} {problem} {problem}.pt data_{idx}.pt \n')
    
    def len(self):
        return sum([len(self.dataset[i].keys()) for i in self.dataset.keys()]) 
    
    def get(self, idx):

        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
        



