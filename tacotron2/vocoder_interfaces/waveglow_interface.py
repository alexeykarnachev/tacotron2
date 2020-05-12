from tacotron2.models.data_model import EncoderOutputWithSpectrogram, EncoderOutput
from tacotron2.vocoder_interfaces._vocoder import Vocoder
from waveglow.models import WaveGlow as WaveGlowModel

from tacotron2.vocoder_interfaces.denoiser import Denoiser


class WrongEncoderOutputTypeError(Exception):
    pass


class WaveGlow(WaveGlowModel, Vocoder):
    def __init__(self, hparams):
        self.hparams = hparams
        super().__init__(self.hparams)
        self.denoiser = None

    @staticmethod
    def _check_encoder_output(encoder_output: EncoderOutput):
        if not isinstance(encoder_output, EncoderOutputWithSpectrogram):
            raise WrongEncoderOutputTypeError("Waveglow is only able to work with spectrogram-like input " +
                                              "Class: EncoderOutputWithSpectrogram")

    def infer(self, encoder_output: EncoderOutputWithSpectrogram, **kwargs):
        self._check_encoder_output(encoder_output)
        if not self.denoiser:
            self.denoiser = Denoiser(self, device=next(self.parameters()).device)

        mel_output_postnet = encoder_output.spectrogram
        sigma = kwargs['sigma']
        denoiser_strength = kwargs['denoiser_strength']

        audio = super().infer(mel_output_postnet, sigma=sigma)
        audio = self.denoiser(audio, strength=denoiser_strength)[:, 0]
        return audio
