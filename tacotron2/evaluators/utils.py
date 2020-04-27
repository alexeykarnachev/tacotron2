import matplotlib.pylab as plt

import IPython.display as ipd
import numpy as np
import torch

from rnd_utilities import get_object

from tacotron2.factory import Factory
from tacotron2.evaluators import BaseEvaluator
from tacotron2.hparams import HParams
from tacotron2.vocoders.denoiser import Denoiser


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
                  encoder_params: dict,
                  vocoder_params: dict,
                  use_denoiser: bool = True,
                  device: str = 'cpu') -> BaseEvaluator:
    """
    Function for creation instance of Evaluator for syntesis
    Args:
        evaluator_classname: `str` class of evaluator
        encoder_params: `Dict` with encoder meta (model, hparams_path, checkpoint_path)
        vocoder_params: `Dict` with vocoder meta (model, hparams_path, checkpoint_path)
        use_denoiser: `bool` use or not postprocessing denoising
        device: `str` identifier for device to use

    Returns:
        `BaseEvaluator` instance
    """

    encoder_model_class = encoder_params['model']
    encoder_hparams = HParams.from_yaml(encoder_params['hparams_path'])
    encoder_hparams.n_symbols = 152
    encoder = get_object(f"tacotron2.models.{encoder_model_class}", encoder_hparams)

    # TODO: Think: is there a chance to make it more simple?
    encoder_weights = torch.load(encoder_params['checkpoint_path'], map_location=device)
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

    vocoder_model_class = vocoder_params['model']
    vocoder_kwargs = {k: v for k, v in vocoder_params.items() if k != 'model'}
    vocoder_kwargs.update({'device': device})
    vocoder = Factory.get_object(f"tacotron2.vocoders.{vocoder_model_class}", **vocoder_kwargs)

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
