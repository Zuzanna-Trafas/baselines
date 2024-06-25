import torch
import h5py
import numpy as np

from torch.utils.data import Dataset
MIMIC_H5_PATH = '/mnt/data/MIMIC-CXR-JPG'


class MIMICDataset(Dataset):
    def __init__(self, split, rgb=False, as_dict=True):
        self.rgb = rgb
        self.hdf5_file_path = f'{MIMIC_H5_PATH}/{split}_224.h5'
        self._load_hdf5_file()

    def _load_hdf5_file(self):
        # Open the HDF5 file
        self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')
        
        # Get the datasets
        self.images_dataset = self.hdf5_file['images']
        self.labels_dataset = self.hdf5_file['labels']
        
        # Calculate the length of the dataset
        self.length = len(self.images_dataset)

    def __getitem__(self, idx):
        if not hasattr(self, 'hdf5_file'):
            self._load_hdf5_file()
            
        image = self.images_dataset[idx]
        if self.rgb:
            image = np.stack([image] * 3, axis=0)
        label = self.labels_dataset[idx]
        
        # Convert label to tensor
        label = torch.FloatTensor(label)
        
        return {
            'image': image, 
            'label': label,
        }

    def __len__(self):
        return self.length