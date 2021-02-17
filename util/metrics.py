import torch.nn as nn
import torch
import pytorch_lightning as pl


class Perplexity(pl.metrics.Metric):
    def __init__(
        self,
        padding_index = int,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group = None,
        dist_sync_fn = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("loss_sum", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.loss = nn.CrossEntropyLoss(ignore_index=padding_index, reduction='sum')
        self.padding_index = padding_index

    @staticmethod
    def match_device(model_out, metric_state):
        metric_state = metric_state.to(model_out.device)
        return metric_state

    def update(self, out, inp):
        y_hat, y = out['logits'], inp['label']
        if y_hat.device != self.loss_sum.device:
            self.loss_sum = self.match_device(y_hat, self.loss_sum)
            self.total = self.match_device(y_hat, self.total)

        if len(y_hat.size()) !=2:
            y_hat = y_hat.contiguous().view(-1, y_hat.size(-1))
            y = y.contiguous().view(-1)
        y = y.contiguous()
        loss = self.loss(y_hat, y)
        self.loss_sum += loss
        self.total += (y != self.padding_index).sum()

    def compute(self):
        """
        Computes accuracy over state.
        """
        return torch.exp(self.loss_sum / self.total)




class TestLoss(pl.metrics.Metric):
    def __init__(
        self,
        padding_index = int,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group = None,
        dist_sync_fn = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("loss_sum", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.padding_index = padding_index

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """

        assert preds.shape == target.shape
        loss = self.loss(preds, target)
        self.loss_sum += loss
        self.total += target.numel()

    def compute(self):
        """
        Computes accuracy over state.
        """
        return torch.exp(self.loss_sum / self.total)
