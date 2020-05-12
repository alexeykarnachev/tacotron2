from dataclasses import dataclass

import torch


@dataclass
class EncoderOutput:
    pass


# TODO: make spectrogram abstract
@dataclass
class EncoderOutputWithSpectrogram(EncoderOutput):
    spectrogram: torch.Tensor


@dataclass
class Tacotron2Output(EncoderOutputWithSpectrogram):
    mel_outputs: torch.Tensor
    spectrogram: torch.Tensor
    gate_outputs: torch.Tensor
    alignments: torch.Tensor
