import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    class MultiTaskLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc1 = nn.Linear(hidden_size, 32)
            self.relu = nn.ReLU()
            self.out_return_1d = nn.Linear(32, 1)  #регрессия
            self.out_direction_1d = nn.Linear(32, 1)  # классификация
            self.out_return_20d = nn.Linear(32, 1)  # регрессия
            self.out_direction_20d = nn.Linear(32, 1)  # классификация
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, (hn, cn) = self.lstm(x)
            out = out[:, -1, :]
            out = self.dropout(out)
            out = self.fc1(out)
            out = self.relu(out)

            return {
                'target_return_1d': self.out_return_1d(out),
                'target_direction_1d': self.sigmoid(self.out_direction_1d(out)),
                'target_return_20d': self.out_return_20d(out),
                'target_direction_20d': self.sigmoid(self.out_direction_20d(out))
            }