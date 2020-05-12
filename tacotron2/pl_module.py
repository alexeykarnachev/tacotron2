from typing import Union
import argparse

import numpy as np
import torch
import pytorch_lightning as pl

from transformers import AdamW
import rnd_datasets

from tacotron2.factory import Factory
from tacotron2.hparams import serialize_hparams
from tacotron2.models import Tacotron2
from tacotron2.models.tacotron2 import Tacotron2KD
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

    def training_epoch_end(self, outputs):
        # Free running `dropout`
        epoch_num = self.trainer.current_epoch
        if 'free_running_rate' in self.hparams:
            if str(epoch_num) in self.hparams['free_running_rate']:
                self.model.decoder.free_running_rate = self.hparams['free_running_rate'][str(epoch_num)]

        return {'log': {'FreeRunningRate': self.model.decoder.free_running_rate}}

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
        idx_batch = np.random.randint(0, len(alignments) - 1)
        idx_inside_batch = np.random.randint(0, self.hparams.batch_size - 1)
        self.logger.experiment.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx_batch][idx_inside_batch, :, :].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.logger.experiment.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx_batch][idx_inside_batch, :, :].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.logger.experiment.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx_batch][idx_inside_batch, :, :].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.logger.experiment.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx_batch][idx_inside_batch].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx_batch][idx_inside_batch]).data.cpu().numpy()),
            iteration, dataformats='HWC')


class TacotronModuleKD(TacotronModule):
    def __init__(self, hparams: Union[dict, argparse.Namespace]):
        super(TacotronModuleKD, self).__init__(hparams)
        self.hparams = hparams

        self._train_dataloader, self._valid_dataloader = prepare_dataloaders(self.hparams)
        self.backbone: Tacotron2 = Factory.get_class(f'tacotron2.models.{hparams.model_class_name}')(hparams)

        # # TODO: load weights incapsulate
        weights = torch.load(hparams['teacher_checkpoint'], map_location='cpu')
        if 'model_state_dict' in weights:
            key_weights_encoder = 'model_state_dict'
        elif 'state_dict' in weights:
            key_weights_encoder = 'state_dict'
        else:
            raise Exception(
                'Cannot take state dict in checkpoint file. Has to have model_state_dict or state_dict key.')

        encoder_weights = weights[key_weights_encoder]
        encoder_weights = {k.split('model.')[-1]: v for k, v in encoder_weights.items()}
        self.backbone.load_state_dict(encoder_weights)

        self.model = Tacotron2KD(self.backbone, self.hparams.get('kd_loss_lambda', 0.))
        self.hparams = serialize_hparams(hparams)

    def training_step(self, batch, batch_idx):
        outputs, loss, loss_mel_student, loss_mel_teacher, loss_kd = self.model(batch)
        lr = self.trainer.optimizers[0].param_groups[-1]['lr']

        logs = {
            'LossOverall/Train': loss,
            'LossMelTeacher/Train': loss_mel_teacher,
            'LossMelStudent/Train': loss_mel_student,
            'LossKD/Train': loss_kd,
            'LearningRate': lr
        }
        to_return = {'loss': loss, 'log': logs}
        if self.trainer.global_step == 0:
            to_return.update({'val_loss': np.inf})

        return to_return

    def validation_step(self, batch, batch_idx):
        outputs, loss, loss_mel_student, loss_mel_teacher, loss_kd = self.model(batch)
        val_gt = batch['y']
        to_return = {
            'val_loss': loss,
            'val_mel_teacher_loss': loss_mel_teacher,
            'val_mel_student_loss': loss_mel_student,
            'val_kd_loss': loss_kd,
            'val_gt': val_gt,
            'val_outputs': outputs
        }
        return to_return

    def validation_epoch_end(self, outputs):
        # Loss to be minimized
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_loss_mel_teacher = torch.stack([x['val_mel_teacher_loss'] for x in outputs]).mean()
        avg_loss_mel_student = torch.stack([x['val_mel_student_loss'] for x in outputs]).mean()
        avg_loss_kd = torch.stack([x['val_kd_loss'] for x in outputs]).mean()
        logs = {
            'Loss/Valid': avg_loss,
            'LossMelTeacher/Valid': avg_loss_mel_teacher,
            'LossMelStudent/Valid': avg_loss_mel_student,
            'LossKD/Valid': avg_loss_kd
        }

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

    def configure_optimizers(self):
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
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
