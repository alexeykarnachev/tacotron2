import random

import torch
from torch.utils.tensorboard import SummaryWriter

from tacotron2.callbacks._callback import Callback
from tacotron2.learner import Learner
from tacotron2.plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from tacotron2.plotting_utils import plot_gate_outputs_to_numpy


class TensorBoardLoggerCallback(Callback):

    def __init__(self, summary_writer: SummaryWriter):
        self.summary_writer = summary_writer

    def on_opt_step(self, learner):
        learning_rate = learner.optimizer.param_groups[0]['lr']

        self.summary_writer.add_scalar("training.loss", learner.train_loss, learner.overall_step)
        self.summary_writer.add_scalar("grad.norm", learner.grad_norm, learner.overall_step)
        self.summary_writer.add_scalar("learning.rate", learning_rate, learner.overall_step)

    def on_eval_end(self, learner: Learner):
        self.summary_writer.add_scalar("validation.loss", learner.valid_loss, learner.overall_step)

        random_batch_id = random.randint(0, len(learner.y_valid_batches) - 1)

        _, mel_outputs, gate_outputs, alignments = learner.y_valid_pred_batches[random_batch_id]
        mel_targets, gate_targets = learner.y_valid_batches[random_batch_id]

        # plot distribution of parameters
        for tag, value in learner.model.named_parameters():
            tag = tag.replace('.', '/')
            self.summary_writer.add_histogram(tag, value.data.cpu().numpy(), learner.overall_step)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.summary_writer.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            learner.overall_step, dataformats='HWC')
        self.summary_writer.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            learner.overall_step, dataformats='HWC')
        self.summary_writer.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            learner.overall_step, dataformats='HWC')
        self.summary_writer.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            learner.overall_step, dataformats='HWC')
