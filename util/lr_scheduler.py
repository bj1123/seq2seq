import math
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

class WarmupExponentialSchedule(LambdaLR):
    """ Linear warmup and then exponential decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Exponentially decreases learning rate.
    """
    def __init__(self, optimizer, warmup_steps, t_epoch,
                 first_decay_ratio=0.1, remainder_decay_ratio=0.75, decay_on_valid=False, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_epoch = t_epoch
        self.coef = 1
        self.gamma = None
        self.first_decay_ratio = first_decay_ratio
        self.remainder_decay_ratio = remainder_decay_ratio
        self.decay_on_valid = decay_on_valid
        super(WarmupExponentialSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        elif step == self.warmup_steps:
            self.gamma = self.get_gamma(self.t_epoch - self.warmup_steps, self.first_decay_ratio)
        elif not step % self.t_epoch:
            self.gamma = self.get_gamma(self.t_epoch, self.remainder_decay_ratio)
        self.coef = self.coef * self.gamma
        return self.coef

    @staticmethod
    def get_gamma(step, target_ratio):
        return math.exp(math.log(target_ratio) / step)


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, decay_on_valid=True, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.decay_on_valid = decay_on_valid
        if self.decay_on_valid:
            self.before_loss = 1e9
            self.decay_coef = 1
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        lrmbda = max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))
        if self.decay_on_valid:
            lrmbda *= self.decay_coef
        return lrmbda

    def decay_baselr(self, val_loss):
        if val_loss > self.before_loss:
            self.decay_coef *= 0.5
        self.before_loss = val_loss


class DecayLearningRate(pl.callbacks.Callback):
    def __init__(self, warmup_steps, t_total, decay_on_valid=False,):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.decay_on_valid = decay_on_valid
        if self.decay_on_valid:
            self.before_loss = 1e9
            self.decay_coef = 1
        self.step = 0
        self.init_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = [param_group['lr'] for param_group in optimizer.param_groups]
            self.init_lrs.append(group)

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        self.step+=1
        if self.step < self.warmup_steps:
            lr_lambda = float(self.step) / float(max(1, self.warmup_steps))
        else:
            lr_lambda = max(0.0, float(self.t_total - self.step) / float(max(1.0, self.t_total - self.warmup_steps)))
        if self.decay_on_valid:
            lr_lambda *= self.decay_coef

        for opt_idx, optimizer in enumerate(trainer.optimizers):
            init_lr_group = self.init_lrs[opt_idx]
            new_lr_group = []
            for p_idx, param_group in enumerate(optimizer.param_groups):
                new_lr = lr_lambda * init_lr_group[p_idx]
                new_lr_group.append(new_lr)
                param_group['lr'] = new_lr