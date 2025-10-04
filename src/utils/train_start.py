from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.loader import Data
from src.models.impl import Model


class Train:
    def __init__(self, data_path: str, device=None, lr=1e-3, epochs=20):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Model(35).to(self.device)
        self.data = Data(path=data_path)
        self.train_loader, self.val_loader = self.data.get_loader()

        self.criterion_reg = nn.MSELoss()
        self.criterion_clf = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs

    def _save_model(self):
        torch.save(self.model, "models/model.csv")

    def fit(self):
        print(f"Training starting on {self.device}")

        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_loss, accuracy_score = self._validate_epoch()

            print(f"Epoch [{epoch+1}/{self.epochs}] | Train loss: {train_loss:.6f} | "
                  f"Val loss: {val_loss:.6f} | Accuracy: {accuracy_score:.6f}")
        self._save_model()

    def _train_epoch(self):
        self.model.train()
        total_loss = 0

        for X_batch, Y_batch in self.train_loader:
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
            preds = self.model(X_batch)

            self.optimizer.zero_grad()
            loss_return_1d = self.criterion_reg(preds['target_return_1d'], Y_batch[:, 0:1])
            loss_direction_1d = self.criterion_clf(preds['target_direction_1d'], Y_batch[:, 1:2])
            loss_return_20d = self.criterion_reg(preds['target_return_20d'], Y_batch[:, 2:3])
            loss_direction_20d = self.criterion_clf(preds['target_direction_20d'], Y_batch[:, 3:4])

            loss = loss_return_1d + loss_return_20d + 0.5*(loss_direction_1d + loss_direction_20d)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            total_correct = 0
            total = 0
            for X_batch, Y_batch in self.val_loader:
                print(len(X_batch), len(Y_batch))
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                preds = self.model(X_batch)
                loss_return_1d = self.criterion_reg(preds['target_return_1d'], Y_batch[:, 0:1])
                loss_direction_1d = self.criterion_clf(preds['target_direction_1d'], Y_batch[:, 1:2])
                loss_return_20d = self.criterion_reg(preds['target_return_20d'], Y_batch[:, 2:3])
                loss_direction_20d = self.criterion_clf(preds['target_direction_20d'], Y_batch[:, 3:4])

                # Общий loss (можно просто суммой или со своими весами)
                loss = loss_return_1d + loss_direction_1d + loss_return_20d + loss_direction_20d
                total_loss += loss.item()
                preds_direction_1d = (preds['target_direction_1d'] > 0.5).float()
                preds_direction_20d = (preds['target_direction_20d'] > 0.5).float()

                total_correct += (preds_direction_1d == Y_batch[:, 1:2]).sum().item()
                total += Y_batch.size(0)
            accuracy = total_correct / total

        return total_loss / len(self.val_loader), accuracy

