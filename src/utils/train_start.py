from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from src.utils.loader import Data
from src.models.impl import Model

class Train:
    def __init__(self, data_path: str, device=None, lr=0.0001, epochs=50):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(34).to(self.device)
        self.data = Data(path=data_path)
        self.train_loader, self.val_loader = self.data.get_loader(seq_len=10, batch_size=16)

        self.criterion_reg = nn.MSELoss()
        self.criterion_clf = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.epochs = epochs
        self.writer = SummaryWriter("src/data/TB_log")

    def _save_model(self, epoch):
        torch.save(self.model.state_dict(), f"models/{epoch}.pth")

    def fit(self):
        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_loss, accuracy = self._validate_epoch()
            print(f"Epoch [{epoch+1}/{self.epochs}] | "
                  f"Train loss: {train_loss:.6f} | "
                  f"Val loss: {val_loss:.6f} | "
                  f"Accuracy: {accuracy:.6f}")
            self.writer.add_scalar("accuracy/epoch", accuracy, epoch)
            self.writer.add_scalar("val_loss/epoch", val_loss, epoch)

        self.writer.close()

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        for X_batch, Y_batch in self.train_loader:
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

            preds = self.model(X_batch)  # [batch, seq_len, 4]

            r1d_pred = preds[:, :, 0]
            d1d_pred = preds[:, :, 1]
            r20d_pred = preds[:, :, 2]
            d20d_pred = preds[:, :, 3]

            r1d_true = Y_batch[:, :, 0]
            d1d_true = Y_batch[:, :, 1]
            r20d_true = Y_batch[:, :, 2]
            d20d_true = Y_batch[:, :, 3]

            loss_return_1d = self.criterion_reg(r1d_pred, r1d_true)
            loss_direction_1d = self.criterion_clf(d1d_pred, d1d_true)
            loss_return_20d = self.criterion_reg(r20d_pred, r20d_true)
            loss_direction_20d = self.criterion_clf(d20d_pred, d20d_true)

            loss = (loss_return_1d + loss_direction_1d + loss_return_20d + loss_direction_20d) / 4

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0
        all_true, all_pred = [], []

        with torch.no_grad():
            for X_batch, Y_batch in self.val_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                preds = self.model(X_batch)  # [batch, seq_len, 4]

                r1d_pred = preds[:, :, 0]
                d1d_pred = preds[:, :, 1]
                r20d_pred = preds[:, :, 2]
                d20d_pred = preds[:, :, 3]

                r1d_true = Y_batch[:, :, 0]
                d1d_true = Y_batch[:, :, 1]
                r20d_true = Y_batch[:, :, 2]
                d20d_true = Y_batch[:, :, 3]

                loss_return_1d = self.criterion_reg(r1d_pred, r1d_true)
                # print(d1d_pred, d1d_true)
                loss_direction_1d = self.criterion_clf(d1d_pred, d1d_true)
                loss_return_20d = self.criterion_reg(r20d_pred, r20d_true)
                loss_direction_20d = self.criterion_clf(d20d_pred, d20d_true)

                loss = (loss_return_1d + loss_direction_1d + loss_return_20d + loss_direction_20d) / 4
                total_loss += loss.item()

                # Для accuracy бинарной классификации
                preds_bin_1d = (d1d_pred > 0.5).int().cpu().numpy().flatten()
                preds_bin_20d = (d20d_pred > 0.5).int().cpu().numpy().flatten()
                true_bin_1d = d1d_true.int().cpu().numpy().flatten()
                true_bin_20d = d20d_true.int().cpu().numpy().flatten()

                all_true.extend(list(true_bin_1d) + list(true_bin_20d))
                all_pred.extend(list(preds_bin_1d) + list(preds_bin_20d))

        accuracy = accuracy_score(all_true, all_pred)

        return total_loss / len(self.val_loader), accuracy
