import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()

        # Output heads
        self.out_return_1d = nn.Linear(64, 1)      # регрессия
        self.out_direction_1d = nn.Linear(64, 1)   # классификация
        self.out_return_20d = nn.Linear(64, 1)     # регрессия
        self.out_direction_20d = nn.Linear(64, 1)  # классификация
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Берём выход последнего шага
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)

        return {
            'target_return_1d': self.out_return_1d(out),
            'target_direction_1d': self.sigmoid(self.out_direction_1d(out)),
            'target_return_20d': self.out_return_20d(out),
            'target_direction_20d': self.sigmoid(self.out_direction_20d(out))
        }

