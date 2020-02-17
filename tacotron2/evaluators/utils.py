import matplotlib.pylab as plt

import IPython.display as ipd
import numpy as np
import torch

from tacotron2.factory import Factory
from tacotron2.hparams import HParams
from tacotron2.evaluators import BaseEvaluator
from waveglow.denoiser import Denoiser


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


def get_evaluator(evaluator_classname: str,
                  encoder_hparams: HParams,
                  encoder_checkpoint_path: str,
                  vocoder_hparams: HParams,
                  vocoder_checkpoint_path: str,
                  use_denoiser: bool = True,
                  device: str = 'cpu') -> BaseEvaluator:
    """
    Function for creation instance of Evaluator for syntesis
    Args:
        evaluator_classname: `str` class of evaluator
        encoder_hparams: `HParams` with tacotron2 meta
        encoder_checkpoint_path: `str` path to tacotron2 checkpoint
        vocoder_hparams: `HParams` with waveglow meta
        vocoder_checkpoint_path: `str` path to waveglow checkpoint
        use_denoiser: `bool` use or not postprocessing denoising
        device: `str` identifier for device to use

    Returns:
        `BaseEvaluator` instance
    """
    encoder = Factory.get_object(f"tacotron2.models.{encoder_hparams['model_class_name']}", encoder_hparams)
    encoder.load_state_dict(
        torch.load(encoder_checkpoint_path, map_location=device)['model_state_dict']
    )

    vocoder = Factory.get_object(f"waveglow.models.{vocoder_hparams['model_class_name']}", vocoder_hparams)
    vocoder.load_state_dict(
        torch.load(vocoder_checkpoint_path, map_location=device)['model_state_dict']
    )

    if use_denoiser:
        denoiser = Denoiser(vocoder, device=device)
    else:
        denoiser = None

    tokenizer = Factory.get_object(f"tacotron2.tokenizers.{encoder_hparams['tokenizer_class_name']}")

    evaluator = Factory.get_object(
        f"tacotron2.evaluators.{evaluator_classname}",
        encoder=encoder,
        vocoder=vocoder,
        tokenizer=tokenizer,
        denoiser=denoiser,
        device=device)

    return evaluator
