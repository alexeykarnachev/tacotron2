import matplotlib.pylab as plt

import IPython.display as ipd
import numpy as np
import torch

from rnd_utilities import get_object

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
    encoder = get_object(f"tacotron2.models.{encoder_hparams['model_class_name']}", encoder_hparams)

    # TODO: Think: is there a chance to make it more simple?
    encoder_weights = torch.load(encoder_checkpoint_path, map_location=device)
    if 'model_state_dict' in encoder_weights:
        key_weights_encoder = 'model_state_dict'
    elif 'state_dict' in encoder_weights:
        key_weights_encoder = 'state_dict'
    else:
        raise Exception('Cannot take state dict in checkpoint file. Has to have model_state_dict or state_dict key.')

    encoder_weights = encoder_weights[key_weights_encoder]
    encoder_weights = {k.split('model.')[-1]: v for k, v in encoder_weights.items()}

    encoder.load_state_dict(encoder_weights)
    encoder.to(device)

    vocoder = Factory.get_object(f"waveglow.models.{vocoder_hparams['model_class_name']}", vocoder_hparams)
    vocoder_loaded_weights = torch.load(vocoder_checkpoint_path, map_location=device)

    if 'model_state_dict' in vocoder_loaded_weights:
        vocoder.load_state_dict(
            torch.load(vocoder_checkpoint_path, map_location=device)['model_state_dict']
        )
    else:
        vocoder.load_state_dict(
            torch.load(vocoder_checkpoint_path, map_location=device)
        )
    vocoder.to(device)

    if use_denoiser:
        denoiser = Denoiser(vocoder, device=device)
    else:
        denoiser = None

    tokenizer = get_object(f"tacotron2.tokenizers.{encoder_hparams['tokenizer_class_name']}")

    evaluator = get_object(
        f"tacotron2.evaluators.{evaluator_classname}",
        encoder=encoder,
        vocoder=vocoder,
        tokenizer=tokenizer,
        denoiser=denoiser,
        device=device)

    return evaluator
