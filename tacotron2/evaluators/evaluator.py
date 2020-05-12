from string import punctuation
from typing import List, Union, Optional, Mapping

import torch
from rnd_utilities import get_object

from tacotron2 import tokenizers

DENOISER_DEFAULT_STRENGTH = 0.03


class Evaluator(object):
    """
    Base evaluator for tacotron + vocoder models pair
    """

    def __init__(self, encoder, vocoder, device='cpu'):

        self.encoder = encoder
        self.vocoder = vocoder
        self.device = device

    @staticmethod
    def _fix_input_text(text):
        # TODO?
        if text.strip(' ')[-1] not in punctuation:
            text = text.strip(' ') + '.'
        return text

    def synthesize(
            self,
            text: Union[str, List[str]],
            encoder_params: Optional[Mapping] = None,
            vocoder_params: Optional[Mapping] = None,
            denoiser_params: Optional[Mapping] = None
    ):
        """
        Args:
            denoiser_params:
            vocoder_params:
            encoder_params:
            text: Text or phonemes input for synthesis.
                  Text can be string, phonemes are List of string phonemes representation + punctuation.
                  Phonemes can be used only with RussianPhonemeTokenizer.
        Returns:
            audio, (mel_outputs_postnet, gates, alignments)
        """

        encoder_params = encoder_params or {}
        vocoder_params = vocoder_params or {}
        denoiser_params = denoiser_params or {}

        text = self._fix_input_text(text)

        with torch.no_grad():
            self.encoder.eval()
            self.vocoder.eval()

            # TODO: make more clear for interfaces: `encoder_output` -> vocoder
            encoder_output = self.encoder.encode_text(text)
            audio = self.vocoder.infer(encoder_output, **vocoder_params)

            return audio, encoder_output
