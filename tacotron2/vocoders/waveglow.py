import torch

from tacotron2.vocoders._vocoder import Vocoder
from tacotron2.hparams import HParams
from waveglow.models import WaveGlow as WaveGlowModel


class WaveGlow(Vocoder):

    def load_state(self):
        hparams_path = self.kwargs['hparams_path']
        hparams = HParams.from_yaml(hparams_path)
        self.model = WaveGlowModel(hparams)

        checkpoint_path = self.kwargs['checkpoint_path']
        weights = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in weights:
            self.model.load_state_dict(
                weights['model_state_dict']
            )
        else:
            self.model.load_state_dict(
                weights
            )
        self.model.to(self.device)

    def infer(self, mel_outputs_postnet, *args, **kwargs):
        sigma = kwargs['sigma']
        audio = self.model.infer(mel_outputs_postnet, sigma=sigma)
        return audio
