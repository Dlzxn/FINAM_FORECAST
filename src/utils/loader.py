
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from src.contracts.data_contract import DataContract


class TorchDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.Y = torch.tensor(Y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Data(DataContract):
    """
    Class implementation of DataContract \n
    >>> path - Path to file with data \n
    >>> train_value = 0.8 - The proportion of data allocated for the training sample
    """

    def __init__(self, path, train_value = 0.8):
        self.train_value = train_value
        self._path = path

    def __len__(self):
        try:
            return len(self.df)
        except AttributeError:
            raise BaseException("DataFrame do not initialized")

    def _load_data(self):
        try:
            self.df = pd.read_csv(self._path)
            self.X = self.df.drop(columns = ["target_return_1d", "target_direction_1d", "target_return_20d",
                                             "target_direction_20d"])

            self.Y = self.df[["target_return_1d", "target_direction_1d", "target_return_20d", "target_direction_20d"]]

            self.train_len = int(len(self.df)*self.train_value)
        except FileNotFoundError:
            raise BaseException("File with data not found")

    def _reset_data(self):
        X_train = self.X[:self.train_len]
        Y_train = self.Y[:self.train_len]

        X_val = self.X[self.train_len:]
        Y_val = self.Y[self.train_len:]

        train_dataset = TorchDataset(X_train, Y_train)
        val_dataset = TorchDataset(X_val, Y_val)

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    def get_loader(self):
        self._load_data()
        self._reset_data()
        return self.train_loader, self.val_loader




