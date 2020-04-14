import numpy as np
import pytest

from tacotron2.audio_preprocessors import preprocessors

@pytest.mark.parametrize(
    "min_, max_",
    [
        (-10, 10),
        (-100, 100),
        (-1000, 1000),
        (-10000, 10000)
    ]
)
def test_normalizer(min_, max_):
    normalizer = preprocessors.AmplitudeNormalizer(min_val=min_, max_val=max_)
    signal = np.random.randn(1000) * max_
    signal_normalized = normalizer.process(signal)

    assert min(signal_normalized) >= min_
    assert max(signal_normalized) <= max_

    corcoeff = np.corrcoef(signal_normalized, signal)
    assert corcoeff.diagonal(1)[0] >= 0.9

@pytest.mark.parametrize(
    "top_db",
    [
        10,
        15,
        30,
        60
    ]
)
def test_trimmer(top_db):
    trimmer = preprocessors.SilenceTrimmer(top_db=top_db, frame_length=16, hop_length=8)
    signal = np.concatenate(
        [
            np.array([0] * 99),
            np.random.randn(100000) * top_db * 10,
            np.array([0] * 123)
        ]
    ).astype('float32')
    len_pre = len(signal)

    signal_normalized = trimmer.process(signal)
    len_post = len(signal_normalized)

    is_subsequence = any(
        [
            np.array_equal(signal[i: i+len_post], signal_normalized)
            for i in range(len_pre - len_post)
        ]
    )

    assert len_pre >= len_post
    assert is_subsequence
