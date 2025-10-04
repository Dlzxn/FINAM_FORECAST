import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TimeDataset(Dataset):
    """
    Dataset для многовыходной регрессии на 20 дней.
    X: признаки
    Y: доходности day_1 .. day_20
    """
    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame, seq_len: int = 20):
        self.seq_len = seq_len
        self.X_seq, self.Y_seq = self.create_sequences(X, Y, seq_len)

    def create_sequences(self, X: pd.DataFrame, Y: pd.DataFrame, seq_len: int):
        Xs, Ys = [], []
        for i in range(len(X) - seq_len + 1):
            Xs.append(X.iloc[i:i + seq_len].values)  # [seq_len, features]
            Ys.append(Y.iloc[i:i + seq_len].values)  # [seq_len, 20] для day_1..day_20
        return torch.tensor(Xs, dtype=torch.float32), torch.tensor(Ys, dtype=torch.float32)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.Y_seq[idx]  # X: [seq_len, features], Y: [seq_len, 20]


class Data:
    """
    Data loader для многовыходной модели.
    """
    def __init__(self, path: str, train_ratio: float = 0.8):
        self._path = path
        self.train_ratio = train_ratio
        self._load_data()

    def _load_data(self):
        self.df = pd.read_csv(self._path)

        # Признаки: все колонки, кроме day_1..day_20
        self.X = self.df.drop(columns=[f"day_{i}" for i in range(1, 21)])

        # Таргеты: day_1..day_20
        self.Y = self.df[[f"day_{i}" for i in range(1, 21)]]

        self.train_len = int(len(self.df) * self.train_ratio)

    def _create_loaders(self, seq_len=20, batch_size=16):
        # Разбивка на train/val
        X_train, Y_train = self.X.iloc[:self.train_len], self.Y.iloc[:self.train_len]
        X_val, Y_val = self.X.iloc[self.train_len:], self.Y.iloc[self.train_len:]

        train_dataset = TimeDataset(X_train, Y_train, seq_len=seq_len)
        val_dataset = TimeDataset(X_val, Y_val, seq_len=seq_len)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def get_loader(self, seq_len=20, batch_size=16):
        self._create_loaders(seq_len=seq_len, batch_size=batch_size)
        return self.train_loader, self.val_loader
