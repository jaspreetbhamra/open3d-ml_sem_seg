import csv, os, sys, pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import pandas as pd
import constants

class Voxels(Dataset):

    def __init__(self, file):
        """
        Args:
            file (string): Path to the pickle file containing data in the form (x,y,z,r,g,b,label).
        """
        with open(file, 'rb') as pickle_file:
            self.data = pickle.load(pickle_file)

        self.coords = self.data[[0, 1, 2]].to_numpy()
        self.pixel_vals = self.data[[3, 4, 5]].to_numpy()
        self.labels = self.data[6].to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return coords, pixel_values, labels
        return self.coords[idx], self.pixel_vals[idx], self.labels[idx]



def get_datasets(config_data):
    train_file_path = os.path.join(sys.path[0], config_data['dataset']['training_file_path'])
    val_file_path = os.path.join(sys.path[0], config_data['dataset']['validation_file_path'])

    train_dataset = Voxels(train_file_path)
    train_data_loader = DataLoader(dataset=train_dataset,
                                    batch_size=config_data['dataset']['batch_size'],
                                    shuffle=False,
                                    num_workers=config_data['dataset']['num_workers'],
                                    pin_memory=False)

    val_dataset = Voxels(val_file_path)
    val_data_loader = DataLoader(dataset=val_dataset,
                                    batch_size=config_data['dataset']['batch_size'],
                                    shuffle=False,
                                    num_workers=config_data['dataset']['num_workers'],
                                    pin_memory=False)
    
    print("Loaded Dataset")
    
    return train_data_loader, val_data_loader