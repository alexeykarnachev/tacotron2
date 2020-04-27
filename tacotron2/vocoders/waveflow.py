import numpy as np
import torch

from tacotron2.utils import load_yaml
from tacotron2.vocoders._vocoder import Vocoder
from waveflow.model import WaveFlow as WaveFlowModel


class WaveFlow(Vocoder):

    def load_state(self):
        hparams_path = self.kwargs['hparams_path']
        hparams = load_yaml(hparams_path)
        self.model = WaveFlowModel(**hparams)

        checkpoint_path = self.kwargs['checkpoint_path']
        weights = torch.load(checkpoint_path, map_location=self.device)
        if 'state_dict' in weights:
            weights['state_dict'] = {k.split('module.')[-1]: v for k, v in weights['state_dict'].items()}
            self.model.load_state_dict(weights['state_dict'])
        else:
            raise RuntimeError('Cannot load checkpoint.')

        self.model.to(self.device)

    def infer(self, mel_outputs_postnet, *args, **kwargs):

        reference = 20
        min_db = -100
        mel_tacotron_like = reference * torch.log10(torch.clamp(mel_outputs_postnet, min=1e-4)) - reference
        mel_tacotron_like = torch.clamp((mel_tacotron_like - min_db) / (-min_db), min=0, max=1)

        sigma = kwargs['sigma']
        audio = self.model.reverse(mel_tacotron_like, temp=sigma)
        return audio
