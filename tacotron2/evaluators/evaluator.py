from string import punctuation
from typing import List, Union

import torch

from tacotron2 import tokenizers

DENOISER_DEFAULT_STRENGTH = 0.03


class BaseEvaluator(object):
    """
    Base evaluator for tacotron + vocoder models pair
    """

    def __init__(self, encoder, vocoder, tokenizer, denoiser=None, device='cpu'):

        self.encoder = encoder
        self.vocoder = vocoder
        self.tokenizer = tokenizer
        self.denoiser = denoiser

        self.device = device

    def synthesize(self, text: Union[str, List[str]], *args, **kwargs):
        """
        Args:
            text: Text or phonemes input for synthesis.
                  Text can be string, phonemes are List of string phonemes representation + punctuation.
                  Phonemes can be used only with RussianPhonemeTokenizer.
        Returns:
            audio, (mel_outputs_postnet, gates, alignments)
        """

        if isinstance(text, list):
            if self.tokenizer.__class__ is not tokenizers.RussianPhonemeTokenizer:
                raise AttributeError("Use Phonemes representation with RussianPhonemeTokenizer only.")

        # This is a hotfix. Have no idea why last sounds disappear in model output.from
        # TODO: Need to figure it out + its better to use re. - like solution (\s+).
        if text.strip(' ')[-1] not in punctuation:
            text = text.strip(' ') + '.'

        with torch.no_grad():
            self.encoder.eval()
            self.vocoder.eval()
            sequence = torch.LongTensor(self.tokenizer.encode(text)) \
                .unsqueeze(0) \
                .to(self.device)

            mel_outputs, mel_outputs_postnet, gates, alignments = self.encoder.inference(sequence)
            audio = self.vocoder.infer(mel_outputs_postnet, sigma=0.9)

            denoiser_strength = kwargs.get('denoiser_strength', DENOISER_DEFAULT_STRENGTH)

            if self.denoiser:
                audio = self.denoiser(audio, strength=denoiser_strength)[:, 0]

            return audio, (mel_outputs_postnet, gates, alignments)


class EmbeddingEvaluator(BaseEvaluator):
    """
    Base evaluator for tacotron + vocoder models pair
    """

    def __init__(self, encoder, vocoder, tokenizer, denoiser=None, device='cpu'):
        super().__init__(encoder, vocoder, tokenizer, denoiser, device)

    def synthesize(self, text, embedding, *args, **kwargs):
        with torch.no_grad():
            self.encoder.eval()
            self.vocoder.eval()
            sequence = torch.LongTensor(self.tokenizer.encode(text)) \
                .unsqueeze(0) \
                .to(self.device)

            embedding = torch.FloatTensor(embedding) \
                .unsqueeze(0) \
                .to(self.device)

            mel_outputs, mel_outputs_postnet, gates, alignments = self.encoder.inference(sequence, embedding)
            audio = self.vocoder.infer(mel_outputs_postnet, sigma=0.9)
            if self.denoiser:
                audio = self.denoiser(audio, strength=0.05)[:, 0]

            return audio, (mel_outputs_postnet, gates, alignments)
