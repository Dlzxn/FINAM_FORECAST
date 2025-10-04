from sklearn.metrics import mean_squared_error, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.utils.loader import Data
from src.models.impl import MultiStepReturnModel  # новая модель

class Train:
    def __init__(self, data_path: str, device=None, lr=0.001, epochs=50):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiStepReturnModel(input_size=33, output_size=20).to(self.device)
        self.data = Data(path=data_path)
        self.train_loader, self.val_loader = self.data.get_loader(seq_len=20, batch_size=16)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.epochs = epochs


    def _save_model(self, epoch):
        torch.save(self.model.state_dict(), f"models/{epoch}.pth")

    def fit(self):
        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_loss, val_rmse, val_acc = self._validate_epoch()
            print(f"Epoch [{epoch+1}/{self.epochs}] | "
                  f"Train loss: {train_loss:.6f} | "
                  f"Val loss: {val_loss:.6f} | "
                  f"Val RMSE: {val_rmse:.6f} | "
                  f"Val Accuracy: {val_acc:.4f}")

            self._save_model(epoch)


    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        for X_batch, Y_batch in self.train_loader:
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

            preds = self.model(X_batch)  # [batch, 20]
            Y_batch = Y_batch[:, -20:, 0]  # [batch, 20]

            loss = self.criterion(preds, Y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0
        all_true, all_pred = [], []
        all_true_dir, all_pred_dir = [], []

        with torch.no_grad():
            for X_batch, Y_batch in self.val_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                preds = self.model(X_batch)  # [batch, 20]
                Y_batch = Y_batch[:, -20:, 0]  # [batch, 20]

                loss = self.criterion(preds, Y_batch)
                total_loss += loss.item()

                # для RMSE
                all_true.extend(Y_batch.cpu().numpy().flatten())
                all_pred.extend(preds.cpu().numpy().flatten())

                # для Accuracy направления
                all_true_dir.extend((Y_batch > 0).cpu().numpy().flatten())
                all_pred_dir.extend((preds > 0).cpu().numpy().flatten())

        val_rmse = mean_squared_error(all_true, all_pred)  # RMSE
        val_acc = accuracy_score(all_true_dir, all_pred_dir)

        return total_loss / len(self.val_loader), val_rmse, val_acc
