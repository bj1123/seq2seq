import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader


class PLWrapper(pl.LightningModule):
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss
        self.val_losses = []

    def training_step(self, batch, batch_idx):
        out = self.model(batch)
        loss = self.loss(out, batch)
        logs = self.loss.to_log(out, batch)
        for key, value in logs.items():
            self.log(key, value, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(batch)
        loss = self.loss(out, batch)
        self.log('val_loss', loss, prog_bar=True)
        logs = self.loss.to_log(out, batch)
        for key, value in logs.items():
            self.log(key + '_val', value, prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        for name, i in self.loss.metrics.items():
            self.log(f'{name}_epoch', i.compute(), prog_bar=True)

    def validation_epoch_end(self, outs):
        for name, i in self.loss.metrics.items():
            name, val = f'{name}_epoch', i.compute()
            self.log(name, val, prog_bar=True)
            self.val_losses.append([name, val])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer


class DataModule(pl.LightningDataModule):
    def __init__(self, training, valid, batch_size):
        super().__init__()
        self.training = training
        self.valid = valid
        self.size = batch_size

    def train_dataloader(self):
        return DataLoader(self.training, batch_size=self.size, collate_fn=self.training.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.size, collate_fn=self.valid.collate_fn)