import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiStepReturnModel(nn.Module):
    """
    Многовыходная модель для прогнозирования доходности на 20 дней вперед.
    Вход: [batch, seq_len=20, input_size]
    Выход: [batch, 20] — доходность для каждого дня
    """

    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3, output_size=20):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size*2, 64)

        self.relu = nn.ReLU()

        # Выход на 20 дней
        self.out_returns = nn.Linear(64, output_size)  # [batch, 20]

    def forward(self, x):
        """
        x: [batch, seq_len, input_size]
        return: [batch, 20]
        """
        out, _ = self.lstm(x)  # [batch, seq_len, hidden]
        # Используем только последний временной шаг для прогноза всего горизонта
        out = out[:, -1, :]  # [batch, hidden]
        out = self.dropout(out)
        out = self.fc1(out)  # [batch, 64]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.out_returns(out)  # [batch, 20]
        return out
