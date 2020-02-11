from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from tacotron2.callbacks._callback import Callback
from tacotron2.models import Tacotron2
from tacotron2.utils import to_device_dict


class Learner:
    def __init__(self, model: Tacotron2, optimizer: Optimizer, callbacks: Optional[List[Callback]]):
        self.model = model
        self.optimizer = optimizer
        self.overall_step = 0
        self.callbacks = callbacks or list()

        self.n_epochs = 0
        self.cur_epoch = 0
        self.n_epoch_steps = 0
        self.cur_epoch_step = 0

        self.train_loss = np.inf
        self.valid_loss = np.inf

        self.device = None
        self.additional_log_fields = dict()

        self.y_valid_batches = None
        self.y_valid_pred_batches = None
        self.grad_norm = None

        self.train_dl = self.valid_dl = None

    def fit(self, dl: Tuple[DataLoader, DataLoader], n_epochs, device, accum_steps, eval_steps, use_all_gpu,
            fp16_opt_level, max_grad_norm):

        self.train_loss = 0
        self.device = device

        self.train_dl, self.valid_dl = dl

        n_gpu = torch.cuda.device_count() if use_all_gpu else 1

        if self.train_dl.batch_size / n_gpu != int(self.train_dl.batch_size / n_gpu):
            raise ValueError(f"You have {n_gpu} GPUs, batch size must be divisible by {n_gpu}")

        self.model = self.model.to(self.device)

        if fp16_opt_level is not None:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=fp16_opt_level)

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.n_epochs = n_epochs
        self.n_epoch_steps = len(self.train_dl)

        pb_epochs = tqdm(range(n_epochs), total=n_epochs, desc='Training')
        [c.on_train_start(learner=self) for c in self.callbacks]

        for cur_epoch in pb_epochs:
            self.cur_epoch = cur_epoch
            self.cur_epoch_step = 0
            pb_epochs.set_postfix({'Epoch': f'{cur_epoch + 1}/{n_epochs}'})
            pb_batches = tqdm(enumerate(self.train_dl), total=len(self.train_dl), desc='Epoch')

            for cur_batch, batch in pb_batches:
                self.cur_epoch_step += 1

                inputs = to_device_dict(batch, device=device)
                self.model.train()
                _, loss = self.model(inputs)

                if n_gpu > 1:
                    loss = loss.mean()

                loss /= accum_steps

                if fp16_opt_level is not None:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.overall_step += 1
                self.train_loss += loss.item()

                if self.overall_step % accum_steps == 0:

                    if fp16_opt_level is not None:
                        self.grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer),
                                                                        max_grad_norm)
                    else:
                        self.grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    self.optimizer.step()
                    [c.on_opt_step(learner=self) for c in self.callbacks]
                    self.optimizer.zero_grad()
                    pb_batches.set_postfix_str(self.get_log_str())
                    self.train_loss = 0

                if (self.overall_step % eval_steps == 0) and self.valid_dl is not None:
                    self.valid_loss = self.eval(dl=self.valid_dl)

                    [c.on_eval_end(learner=self) for c in self.callbacks]

            [c.on_epoch_end(learner=self) for c in self.callbacks]

        [c.on_train_end(learner=self) for c in self.callbacks]

    def eval(self, dl):
        device = next(self.model.parameters()).device

        self.y_valid_batches = []
        self.y_valid_pred_batches = []

        with torch.no_grad():
            pb_valid = tqdm(dl, total=len(dl), desc='Validation')
            valid_loss = 0

            for batch in pb_valid:
                inputs = to_device_dict(batch, device=device)

                self.model.eval()
                y_valid_pred_, loss = self.model(inputs)
                valid_loss += loss.item()

                self.y_valid_batches.append(inputs['y'])
                self.y_valid_pred_batches.append(y_valid_pred_)

            valid_loss /= len(pb_valid)

        return valid_loss

    def get_log_str(self):
        epoch_str = f"Epoch: {self.cur_epoch + 1}/{self.n_epochs}"
        losses_str = f"Loss/Valid: {round(self.valid_loss, 5)}, Loss/Train: {round(self.train_loss, 5)}"
        str_ = ', '.join([epoch_str, losses_str])

        if len(self.additional_log_fields):
            add_str = ", ".join(f'{k}: {v}' for k, v in self.additional_log_fields.items())
            str_ = ', '.join([str_, add_str])
        return str_
