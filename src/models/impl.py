import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Модель для прогнозирования 4 параметров по временным рядам свечей:
    - target_return_1d       (регрессия)
    - target_direction_1d    (бинарная классификация)
    - target_return_20d      (регрессия)
    - target_direction_20d   (бинарная классификация)

    На выходе: [batch, seq_len, 4]
    """

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

        # Головки выхода (4 параметра)
        self.out_return_1d = nn.Linear(64, 1)  # регрессия
        self.out_direction_1d = nn.Linear(64, 1)  # классификация
        self.out_return_20d = nn.Linear(64, 1)  # регрессия
        self.out_direction_20d = nn.Linear(64, 1)  # классификация
        self.dropout = nn.Dropout(0.2) #регуляризация

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [batch, seq_len, input_size]
        return: [batch, seq_len, 4]
        """
        out, _ = self.lstm(x)  # [batch, seq_len, hidden]
        out = self.dropout(out)
        out = self.fc1(out)  # [batch, seq_len, 64]
        out = self.relu(out)
        out = self.dropout(out)

        # Каждая головка делает предсказание для каждого временного шага
        r1d = self.out_return_1d(out)  # [batch, seq_len, 1]
        d1d = self.sigmoid(self.out_direction_1d(out))  # [batch, seq_len, 1]
        r20d = self.out_return_20d(out)  # [batch, seq_len, 1]
        d20d = self.sigmoid(self.out_direction_20d(out))  # [batch, seq_len, 1]

        combined_output = torch.cat([r1d, d1d, r20d, d20d], dim=2)
        return combined_output

