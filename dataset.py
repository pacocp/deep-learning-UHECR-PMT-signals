import torch
from torch.utils import data


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, inputs, labels):
        'Initialization'
        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)
        
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        x = torch.from_numpy(self.inputs[index]).float()
        y = torch.from_numpy(self.labels[index]).float()   
        return x, y


class DatasetExo(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, inputs, inputs_exo, labels):
        'Initialization'
        self.labels = labels
        self.inputs = inputs
        self.inputs_exo = inputs_exo

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)
        
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        x = torch.from_numpy(self.inputs[index]).float()
        x_exo = torch.from_numpy(self.inputs_exo[index]).float()
        y = torch.from_numpy(self.labels[index]).float()   
        return (x, x_exo, y)


class Dataset3TanksSeparated(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, inputs_tank1, inputs_tank2, inputs_tank3, labels):
        'Initialization'
        self.labels = labels
        self.inputs_tank1 = inputs_tank1
        self.inputs_tank2 = inputs_tank2
        self.inputs_tank3 = inputs_tank3

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs_tank1)
     
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        x_tank1 = torch.from_numpy(self.inputs_tank1[index]).float()
        x_tank2 = torch.from_numpy(self.inputs_tank2[index]).float()
        x_tank3 = torch.from_numpy(self.inputs_tank3[index]).float()
        y = torch.from_numpy(self.labels[index]).float()
        return (x_tank1, x_tank2, x_tank3, y)


class Dataset3TanksSeparatedExo(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, inputs_tank1, inputs_tank2, inputs_tank3, exos, labels):
        'Initialization'
        self.labels = labels
        self.inputs_tank1 = inputs_tank1
        self.inputs_tank2 = inputs_tank2
        self.inputs_tank3 = inputs_tank3
        self.exos = exos

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs_tank1)
     
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        x_tank1 = torch.from_numpy(self.inputs_tank1[index]).float()
        x_tank2 = torch.from_numpy(self.inputs_tank2[index]).float()
        x_tank3 = torch.from_numpy(self.inputs_tank3[index]).float()
        exos_aux = [torch.from_numpy(self.exos[0][index]).float(), 
                    torch.from_numpy(self.exos[1][index]).float(),
                    torch.from_numpy(self.exos[2][index]).float()]
        y = torch.from_numpy(self.labels[index]).float()
        return (x_tank1, x_tank2, x_tank3, exos_aux, y)