import pytorch_lightning as pl
import torch


class PLWrapper(pl.LightningModule):
    def __init__(self, model, loss, lr):
        super(PLWrapper, self).__init__()
        self.model = model
        self.manual_loss = loss
        self.lr = lr

    def forward(self, inp):
        return self.model(inp)

    def training_step(self, batch, batch_idx):
        out = self.model(batch)
        loss = self.loss(out, batch)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer