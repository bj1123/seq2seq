from tqdm import tqdm
import math
from model.ops import *
import apex
from util.losses import *
from abc import ABC, abstractmethod
import os


def get_acc(logits, y):
    _, predicted = torch.max(logits.data, 1)
    correct = (predicted == y).sum().item()
    return correct / y.numel()


def top_k_acc(logits, y, top_k):
    total = y.size(0)
    _, indices = torch.topk(logits, top_k, 1)
    indices = indices.t()
    correct = indices.eq(y.view(1, -1).expand_as(indices))
    return correct.sum().item(), total


def check_empty_text(x):
    for key in x.keys():
        if 'len' in key and 0 in x[key]:
            return True
    return False


class Trainer:
    def __init__(self, model, train_batchfier, test_batchfier, optimizers, schedulers,
                 update_step, criteria, clip_norm, mixed_precision, **kwargs):
        self.model = model
        self.train_batchfier = train_batchfier
        self.test_batchfier = test_batchfier
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.criteria = criteria
        self.step = 0
        self.epoch = -1
        self.update_step = update_step
        self.mixed_precision = mixed_precision
        self.clip_norm = clip_norm
        self.intermittently_save_ckpt = kwargs.pop('intermittently_save_ckpt', None)
        if self.intermittently_save_ckpt:
            self.n_ckpt = kwargs.pop('n_ckpt', None)
            self.target_cnt = len(self.train_batchfier) // self.n_ckpt
            self.save_name = kwargs.pop('save_name', None)

    def train_epoch(self):
        def reset_pbar(pbar, n_bar):
            criteria.clear_loss()
            pbar.close()
            pbar = tqdm(100)
            return pbar, n_bar + 1, 0, 0, 0
        self.epoch +=1
        model = self.model
        batchfier = self.train_batchfier.to_iterator()
        criteria = self.criteria
        optimizer = self.optimizers
        scheduler = self.schedulers
        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0
        criteria.clear_loss()
        pbar = tqdm(100) # Null pbar
        pbar_cnt = 0
        model.zero_grad()
        cnt = 0
        for inp in batchfier:
            cnt+=1
            if check_empty_text(inp):
                continue
            out = model(inp)
            loss = criteria(out, inp)
            step_loss += loss.item()
            tot_loss += loss.item()
            if self.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # print(model.main_nets[0].bd.v_net.weight.grad)
            # print('--'*50)
            # print(model.main_nets[0].blocks[0].feedforward.net[0].weight.grad)
            tot_cnt += 1

            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                model.zero_grad()
                scheduler.step(self.step)
                description = criteria.get_description(self.update_step * pbar_cnt)
                description = self.update_description(description, n_bar)
                pbar.set_description(description)
                # pbar.set_description(
                #     "training loss : %f training ppl : %f, lr : %f, iter : %d" % (
                #         step_loss / (self.update_step *pbar_cnt), math.exp(step_loss / (self.update_step*pbar_cnt)),
                #          scheduler.get_last_lr()[0], n_bar), )
                pbar.update()
                if pbar_cnt == 100:
                    pbar, n_bar, pbar_cnt, step_loss, acc = reset_pbar(pbar, n_bar)
            if self.intermittently_save_ckpt:
                if not tot_cnt % self.target_cnt:
                    print('save!!!!!!!!!!!!!!!')
                    torch.save(model.state_dict(),
                               os.path.join(self.save_name,'epoch_{}_ckpt_{}'.format(self.epoch,
                                                                                    tot_cnt // self.target_cnt)))


        pbar.close()
        loss = math.exp(tot_loss / tot_cnt)
        return loss

    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier.to_iterator()

        if isinstance(self.criteria,tuple):
            _,criteria= self.criteria
        else:
            criteria = self.criteria
        model.eval()
        criteria.clear_loss()
        pbar = tqdm(batchfier)
        pbar_cnt = 0
        step_loss = 0
        for inp in pbar:
            with torch.no_grad():
                if check_empty_text(inp):
                    continue
                out = model(inp)

                loss = criteria(out['logits'], inp['label'])
                step_loss += loss.item()
                pbar_cnt += 1
                description = criteria.get_description(pbar_cnt)
                pbar.set_description(description)
        pbar.close()
        loss = math.exp(step_loss / pbar_cnt)
        if getattr(self.schedulers, 'decay_on_valid', None):
            self.schedulers.decay_baselr(loss)
        return loss

    def update_description(self, description, n_bar):
        description += 'lr : %f, iter : %d ' % (self.schedulers.get_last_lr()[0], n_bar)
        return description



