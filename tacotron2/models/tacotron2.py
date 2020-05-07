from math import sqrt
from copy import deepcopy

import torch
from torch import nn

from tacotron2.loss_function import Tacotron2Loss, MaskedMSELoss
from tacotron2.models._modules import Encoder, Decoder, Postnet
from tacotron2.utils import get_mask_from_lengths


class Tacotron2(nn.Module):

    @property
    def device(self):
        return self.parameters().__next__().device

    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.hparams = hparams
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

        self.criterion = Tacotron2Loss()

    def parse_output(self, outputs, output_lengths=None):
        # Todo: move to loss fn?
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)

            mask_len = mask.size(1)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask_len)
            mask = mask.permute(1, 0, 2)

            output_len = outputs[0].size(2)
            pad_range = mask_len - min(output_len, mask_len)

            outputs[0] = outputs[0][:, :, :mask_len]
            outputs[0] = torch.nn.functional.pad(outputs[0], pad=(0, pad_range), value=0.0)
            outputs[0].data.masked_fill_(mask, 0.0)

            outputs[1] = outputs[1][:, :, :mask_len]
            outputs[1] = torch.nn.functional.pad(outputs[1], pad=(0, pad_range), value=0.0)
            outputs[1].data.masked_fill_(mask, 0.0)

            outputs[2] = outputs[2][:, :mask_len]
            outputs[2] = torch.nn.functional.pad(outputs[2], pad=(0, pad_range), value=0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

            outputs[3] = outputs[3][:, :mask_len]
            outputs[3] = torch.nn.functional.pad(outputs[3], pad=(0, pad_range), value=0.0)

        return outputs

    def encode(self, text_inputs, text_lengths):
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        return encoder_outputs

    def decode(self, encoder_outputs, mels, text_lengths):
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)
        return mel_outputs, gate_outputs, alignments

    def forward(self, inputs):
        """
        inputs: dict{'x': (...), 'y':(...)}
        """
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs['x']
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        encoder_outputs = self.encode(text_inputs, text_lengths)
        mel_outputs, gate_outputs, alignments = self.decode(encoder_outputs, mels, text_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments], output_lengths)
        loss = self.criterion(outputs, inputs['y'], output_lengths) if inputs['y'] is not None else None

        return outputs, loss

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs


class Tacotron2KD(nn.Module):
    def __init__(self, backbone: Tacotron2, kd_loss_lambda: int):
        super().__init__()

        self.backbone = backbone
        # for param in self.backbone.embedding.parameters():
        #     param.requires_grad = False
        # for param in self.backbone.encoder.parameters():
        #     param.requires_grad = False

        self.student_decoder = deepcopy(self.backbone.decoder)
        # for param in self.backbone.decoder.parameters():
        #     param.requires_grad = False

        self.kd_loss = MaskedMSELoss()
        self.kd_loss_lambda = kd_loss_lambda

    def decode(self, encoder_outputs, mels, text_lengths):
        mel_outputs, gate_outputs, alignments = \
            self.backbone.decode(encoder_outputs, mels, text_lengths)
        mel_outputs_student, gate_outputs_student, alignments_student = \
            self.student_decoder.inference(encoder_outputs)
        gate_outputs_student = gate_outputs_student.squeeze(2)
        return (mel_outputs, gate_outputs, alignments), \
               (mel_outputs_student, gate_outputs_student, alignments_student)

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs['x']
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        encoder_outputs = self.backbone.encode(text_inputs, text_lengths)

        (mel_outputs, gate_outputs, alignments), \
        (mel_outputs_student, gate_outputs_student, alignments_student) = self.decode(
            encoder_outputs, mels, text_lengths)

        mel_outputs_postnet = self.backbone.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        mel_outputs_postnet_student = self.backbone.postnet(mel_outputs_student)
        mel_outputs_postnet_student = mel_outputs_student + mel_outputs_postnet_student

        # TODO: Dataclass of decoder output
        outputs = self.backbone.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths
        )
        outputs_student = self.backbone.parse_output(
            [mel_outputs_student, mel_outputs_postnet_student, gate_outputs_student, alignments_student],
            output_lengths
        )
        loss_mel_student = self.backbone.criterion(outputs_student, inputs['y'], output_lengths)
        loss_mel_teacher = self.backbone.criterion(outputs, inputs['y'], output_lengths)
        loss_kd = self.kd_loss(outputs_student[0], mel_outputs, output_lengths)
        loss = loss_mel_student + loss_mel_teacher + self.kd_loss_lambda * loss_kd

        return outputs_student, loss, loss_mel_student, loss_mel_teacher, loss_kd

    def inference(self, inputs):
        embedded_inputs = self.backbone.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.backbone.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.student_decoder.inference(encoder_outputs)

        mel_outputs_postnet = self.backbone.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        # TODO: we dont need this
        outputs = self.backbone.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
