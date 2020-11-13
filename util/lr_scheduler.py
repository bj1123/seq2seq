import math
from torch.optim.lr_scheduler import LambdaLR


class WarmupExponentialSchedule(LambdaLR):
    """ Linear warmup and then exponential decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Exponentially decreases learning rate.
    """
    def __init__(self, optimizer, warmup_steps, t_epoch,
                 first_decay_ratio=0.1, remainder_decay_ratio=0.75, decay_on_valid=True, last_epoch=-1):
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
        print('decay_coefffffffffffffff')
        if val_loss > self.before_loss:
            self.decay_coef *= 0.5
        self.before_loss = val_loss
