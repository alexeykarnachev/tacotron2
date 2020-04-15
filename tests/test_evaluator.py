import pytest
import torch
import numpy as np

from tacotron2.evaluators import BaseEvaluator
from tacotron2.tokenizers import RussianPhonemeTokenizer


class DummyEncoder(object):
    def eval(self):
        pass

    def inference(self, tokens: torch.LongTensor):
        mel_outputs = tokens.numpy()[0]
        mel_outputs_postnet = mel_outputs + 1
        gates = tokens.min().item()
        alignments = tokens.max().item()

        return mel_outputs, mel_outputs_postnet, gates, alignments


class DummyVocoder(object):
    def eval(self):
        pass

    def infer(self, mel_outputs_postnet: np.ndarray, sigma=0.9):
        signal = ''.join([str(x) for x in mel_outputs_postnet.tolist()])
        return signal


class DummyTokenizer(RussianPhonemeTokenizer):
    def encode(self, text):
        return [len(x) for x in text.split()]

    def __class__(self):
        return RussianPhonemeTokenizer


class DummyDenoiser(object):
    def __call__(self, audio, strength):
        tokens = [int(x) if int(x) < strength else 0 for x in audio]
        output = np.zeros((len(tokens), 2), dtype='int32')
        output[:, 0] = tokens
        return output


@pytest.mark.parametrize(
    [
        "use_denoiser",
        "denoiser_strength",
        "text",
        "expected_result"],
    [
        (True, 99, 'Мороз и солнце день чудесeн.', np.array([6, 2, 7, 5, 9])),
        (True, 6, 'Мороз и солнце день чудесeн.', np.array([0, 2, 0, 5, 0])),
        (False, 99, 'Мороз и солнце день чудесен.', '62759')
    ]
)
def test_evaluator(use_denoiser, denoiser_strength, text, expected_result):
    encoder = DummyEncoder()
    vocoder = DummyVocoder()
    tokenizer = DummyTokenizer()

    if use_denoiser:
        denoiser = DummyDenoiser()
    else:
        denoiser = None

    evaluator = BaseEvaluator(
        encoder=encoder,
        vocoder=vocoder,
        denoiser=denoiser,
        tokenizer=tokenizer,
        device='cpu')

    encoded = evaluator.synthesize(text, denoiser_strength=denoiser_strength)

    if use_denoiser:
        assert np.array_equal(encoded[0], expected_result)
    else:
        assert np.array_equal(encoded[0], expected_result)

    assert max(encoded[1][0]) - 1 == encoded[1][2]
    assert min(encoded[1][0]) - 1 == encoded[1][1]
