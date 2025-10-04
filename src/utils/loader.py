
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from src.contracts.data_contract import DataContract


import torch
from torch.utils.data import Dataset

class TimeDataset(Dataset):
    def __init__(self, X, Y, seq_len=10):
        self.seq_len = seq_len
        self.X_seq, self.Y_seq = self.create_sequences(X, Y, seq_len)

    def create_sequences(self, X, Y, seq_len):
        Xs, Ys = [], []
        for i in range(len(X) - seq_len + 1):
            Xs.append(X.iloc[i:i + seq_len].values)  # окно из seq_len свечей
            Ys.append(Y.iloc[i:i + seq_len].values)  # таргет для каждого шага окна
        return torch.tensor(Xs, dtype=torch.float32), torch.tensor(Ys, dtype=torch.float32)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.Y_seq[idx]





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
                                             "target_direction_20d", "day"])

            self.Y = self.df[["target_return_1d", "target_direction_1d", "target_return_20d", "target_direction_20d"]]

            self.train_len = int(len(self.df)*self.train_value)
        except FileNotFoundError:
            raise BaseException("File with data not found")

    def _reset_data(self, seq_len=10, batch_size=16):
        X_train = self.X[:self.train_len]
        Y_train = self.Y[:self.train_len]

        X_val = self.X[self.train_len:]
        Y_val = self.Y[self.train_len:]

        train_dataset = TimeDataset(X_train, Y_train, seq_len=seq_len)
        val_dataset = TimeDataset(X_val, Y_val, seq_len=seq_len)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def get_loader(self, seq_len=10, batch_size=16):
        self._load_data()
        self._reset_data(seq_len=seq_len, batch_size=batch_size)
        return self.train_loader, self.val_loader





