import pytorch_lightning as pl
import torch.nn as nn


class Perplexity(pl.metrics.Metric):
    def __init__(
        self,
        padding_index = int,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("loss_sum", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.loss = nn.CrossEntropyLoss(ignore_index=padding_index, reduction='sum')
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
        target == self.padding_index
        self.loss_sum += loss
        self.total += (target != self.padding_index).sum()

    def compute(self):
        """
        Computes accuracy over state.
        """
        return torch.exp(self.loss_sum / self.total)
