from sklearn.discriminant_analysis import StandardScaler
import torch
from torch.utils.data import Dataset, random_split
import pandas as pd


class NetworkDataset(Dataset):
    def __init__(self, datapath):
        self.dataframe = pd.read_pickle(datapath)
        self.labels = self.dataframe['Label'].values
        self.features= self.dataframe.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR','L4_SRC_PORT', 'L4_DST_PORT', 'Attack', 'Label'] ).values
        scaler = StandardScaler()  
        self.features = scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def load_dataset(client_id : int, batch_size: int ): 
    dataset_paths = {
        1: "./data/NF-ToN-IoT_preprocessed.pkl",
        2: "./data/NF-ToN-IoT-Modified_preprocessed.pkl",
        3: "./data/NF-BoT-IoT_preprocessed.pkl",
        4: "./data/NF-UNSW-NB15-Modified_preprocessed.pkl"
    }

    client_dataset_path = dataset_paths.get(client_id, None)
    client_dataset = NetworkDataset(client_dataset_path)
    
    train_set, test_set = random_split(client_dataset, [0.7, 0.3])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

