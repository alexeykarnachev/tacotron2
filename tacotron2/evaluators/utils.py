from collections import Mapping
from pathlib import Path

import matplotlib.pylab as plt

import IPython.display as ipd
import numpy as np
import torch

from rnd_utilities import get_object

from tacotron2.evaluators import BaseEvaluator
from tacotron2.hparams import HParams
from tacotron2.vocoder_interfaces._vocoder import Vocoder
from tacotron2.vocoder_interfaces.denoiser import Denoiser


def plot_syntesis_result(data, figsize=(16, 4)):
    """
    Helper to plot syntesis result
    Args:
        data: Union[np.array]  with mel spectrogam, alignment map etc
        figsize: `tuple` with sizes

    Returns:
        plt.figure
    """
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')

    return fig


def jupyter_play_syntesed(audiodata: np.array, sr: int):
    """
    Function to create player in ipynb enviroment
    Args:
        audiodata: `np.array` signal data
        sr: `int` sampling rate

    Returns:

    """
    ipd.Audio(audiodata[0].data.cpu().numpy(), rate=sr)


def get_vocoder(vocoder_params, device) -> Vocoder:
    vocoder_model_class = vocoder_params['model']
    vocoder_kwargs = {k: v for k, v in vocoder_params.items() if k != 'model'}
    vocoder_kwargs.update({'device': device})
    vocoder = get_object(f"tacotron2.vocoders.{vocoder_model_class}", **vocoder_kwargs)
    return vocoder


def clean_parameter_names(weights: Mapping) -> Mapping:
    return {k.split('model.')[-1]: v for k, v in weights.items()}


def get_model_from_checkpoint(ckpt_path, package_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')

    weights = ckpt['state_dict']
    weights = clean_parameter_names(weights)
    weights = {k.split('model.')[-1]: v for k, v in weights.items()}

    hparams_raw = ckpt['hparams']
    hparams = HParams(hparams_raw)

    model = get_object(
        f"{package_path}.{hparams.model_class_name}",
        hparams=hparams
    )
    model.load_state_dict(weights)
    model.to(device)
    return model


def get_encoder_from_checkpoint(ckpt_path, device):
    return get_model_from_checkpoint(ckpt_path, 'tacotron2.models', device)


def get_vocoder_from_checkpoint(ckpt_path, device):
    return get_model_from_checkpoint(ckpt_path, 'tacotron2.vocoder_interfaces', device)


def get_evaluator(evaluator_classname: str,
                  encoder_ckpt_path: Path,
                  vocoder_ckpt_path: Path,
                  device: str = 'cpu'
) -> BaseEvaluator:
    """
    Function for creation instance of Evaluator for syntesis
    Args:
        vocoder_ckpt_path:
        encoder_ckpt_path:
        evaluator_classname: `str` class of evaluator
        device: `str` identifier for device to use

    Returns:
        `BaseEvaluator` instance
    """

    encoder = get_encoder_from_checkpoint(encoder_ckpt_path, device)
    vocoder = get_vocoder_from_checkpoint(vocoder_ckpt_path, device)

    evaluator = get_object(
        f"tacotron2.evaluators.{evaluator_classname}",
        encoder=encoder,
        vocoder=vocoder,
        device=device
    )

    return evaluator
