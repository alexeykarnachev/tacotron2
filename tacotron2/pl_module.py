from typing import Union
import argparse

import numpy as np
import torch
import pytorch_lightning as pl

from transformers import AdamW
import rnd_datasets

from tacotron2.factory import Factory
from tacotron2.hparams import serialize_hparams
from tacotron2.plotting_utils import plot_spectrogram_to_numpy, plot_gate_outputs_to_numpy, plot_alignment_to_numpy
from tacotron2.utils import prepare_dataloaders


class TacotronModule(pl.LightningModule):
    def __init__(self, hparams: Union[dict, argparse.Namespace]):
        super(TacotronModule, self).__init__()

        self.hparams = hparams

        self._train_dataloader, self._valid_dataloader = prepare_dataloaders(self.hparams)
        self.model = Factory.get_class(f'tacotron2.models.{hparams.model_class_name}')(hparams)

        self.hparams = serialize_hparams(hparams)

    def forward(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):

        outputs, loss = self.model(batch)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        logs = {'Loss/Train': loss, 'LearningRate': lr}
        to_return = {'loss': loss, 'log': logs}
        if self.trainer.global_step == 0:
            to_return.update({'val_loss': np.inf})

        return to_return

    def validation_step(self, batch, batch_idx):
        outputs, loss = self.model(batch)
        val_gt = batch['y']
        return {'val_loss': loss, 'val_gt': val_gt, 'val_outputs': outputs}

    def validation_epoch_end(self, outputs):
        # Loss to be minimized
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'Loss/Valid': avg_loss}

        # Ground truth labels for all validation
        val_gt = [x['val_gt'] for x in outputs]
        mel_gt, gate_gt = list(zip(*val_gt))

        # Prediction results for all validation
        val_outputs = [x['val_outputs'] for x in outputs]
        _, mel_outputs_postnet, gate_outputs, alignments = list(zip(*val_outputs))

        # Logging of additional validation data (alignments, gates etc).
        iteration = self.trainer.global_step
        self.log_validation_results(
            y=[mel_gt, gate_gt],
            y_pred=[mel_outputs_postnet, gate_outputs, alignments],
            iteration=iteration
        )

        return {'val_loss': avg_loss, 'log': logs}

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._valid_dataloader

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay)

        lr_scheduler = rnd_datasets.ReduceOnPlateauWithWarmup(
            optimizer=optimizer,
            warmup_steps=self.hparams.warmup_steps,
            factor=self.hparams.lr_reduce_factor,
            patience=self.hparams.lr_reduce_patience,
            global_step=self.trainer.global_step
        )

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': self.trainer.accumulate_grad_batches,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }

        return [optimizer], [scheduler]

    def check_grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def log_validation_results(self, y, y_pred, iteration):

        # Outputs logging
        mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in self.model.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.experiment.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = np.random.randint(0, alignments.size(0) - 1)
        self.logger.experiment.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.logger.experiment.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.logger.experiment.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.logger.experiment.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
