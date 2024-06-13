import torch
import h5py

from torch.utils.data import Dataset
from constants import *


class PretrainDataset(Dataset):
    def __init__(self, split, transform=None):
        self.hdf5_file_path = f'{MIMIC_H5_PATH}/{split}.h5'
        self.transform = transform
        
        # Open the HDF5 file
        self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')
        
        # Get the datasets
        self.images_dataset = self.hdf5_file['images']
        self.labels_dataset = self.hdf5_file['labels']
        self.reports_dataset = self.hdf5_file['reports']
        
        # Calculate the length of the dataset
        self.length = len(self.images_dataset)

    def __getitem__(self, idx):
        image = self.images_dataset[idx]
        label = self.labels_dataset[idx]
        report = self.reports_dataset[idx]
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        # Convert label to tensor
        label = torch.FloatTensor(label)
        
        return {
            'image': image, 
            'label': label,
            'report': report
        }

    def __len__(self):
        return self.length