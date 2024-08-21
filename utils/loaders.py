import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io

class SigGenDataset(Dataset):
    def __init__(self, filepaths : list, transform = True, phase = None):
        """
        Initialize the dataset with a file path and delimiter.
        
        Parameters:
        - file_path: list of str.
        """
        
        arrays_to_append_data = []
        arrays_to_append_labels = []

        self.phase = phase
        
        self.ind_start = 0
        
        for i in range(len(filepaths)):
            mat = scipy.io.loadmat(filepaths[i])
            df = mat['X']
            
            if transform:
                df = df.T
            arrays_to_append_data.append(df)
            arrays_to_append_labels.append(np.ones((df.shape[0]), dtype = np.int8)*(i % 4))
            
            
        
        self.data = np.vstack(arrays_to_append_data)
        if transform:
            self.data = np.expand_dims(self.data, axis=2)
        else:
            self.data = np.expand_dims(self.data, axis=1)

        self.labels = np.concatenate(arrays_to_append_labels)
        
        if phase == "test":
            self.ind_start = int(len(self) * 0.8)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        if self.phase == None:
            return self.data.shape[0]
        elif self.phase == "train":
            return int(self.data.shape[0] * 0.8)
        elif self.phase == "test":
            return int(self.data.shape[0] * 0.2)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.
        
        Parameters:
        - idx: int, index of the sample.
        
        Returns:
        - tuple (features, label)
        """
        features = self.data[idx + self.ind_start]
        label = self.labels[idx + self.ind_start]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class SigGenDatasetSplit(Dataset):
    def __init__(self, filepaths : list, transform = True, phase = None, means = None):
        """
        Initialize the dataset with a file path and delimiter.
        
        Parameters:
        - file_path: list of str.
        """
        
        arrays_to_append_data = []
        arrays_to_append_labels = []
        
        self.phase = phase
        self.means = means
        
        self.ind_start = 0
        
        for i in range(len(filepaths)):
            mat = scipy.io.loadmat(filepaths[i])
            df = mat['X']
            
            if transform:
                df = df.T
            
            if phase == "train":
                df_split = df[:int(df.shape[0] * 0.8)]
            else:
                df_split = df[int(df.shape[0] * 0.8):]
                
                
            arrays_to_append_data.append(df_split)
            arrays_to_append_labels.append(np.ones((df_split.shape[0]), dtype = np.int8)*(i % 4))
            
            
        
        self.data = np.vstack(arrays_to_append_data)
        if transform:
            self.data = np.expand_dims(self.data, axis=2)
        else:
            self.data = np.expand_dims(self.data, axis=1)

        self.labels = np.concatenate(arrays_to_append_labels)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.
        
        Parameters:
        - idx: int, index of the sample.
        
        Returns:
        - tuple (features, label)
        """
        features = self.data[idx]
        label = self.labels[idx]
        
        if self.means == None:
            return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        else:
            return torch.tensor(features, dtype=torch.float32), torch.tensor(self.means[label], dtype=torch.float32)