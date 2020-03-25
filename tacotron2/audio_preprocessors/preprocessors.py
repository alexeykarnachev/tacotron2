import librosa
import numpy as np

from . import _audio_preprocessor


class SilenceTrimmer(_audio_preprocessor.AudioPreprocessor):
    """Audio processor which trims silence parts on the beginning and on the end of input audio signal"""

    def __init__(self, top_db: int = 30, frame_length: int = 512, hop_length: int = 256):
        """
        Args:
            top_db: number > 0
                The threshold (in decibels) below reference to consider as
                silence

            frame_length: int > 0
                The number of samples per analysis frame

            hop_length: int > 0
                The number of samples between analysis frames
        """
        self._top_db = top_db
        self._frame_length = frame_length
        self._hop_length = hop_length

    def process(self, audio: np.ndarray) -> np.ndarray:
        data, _ = librosa.effects.trim(
            audio.astype(np.float32), top_db=self._top_db, frame_length=self._frame_length, hop_length=self._hop_length
        )

        return data


class AmplitudeNormalizer(_audio_preprocessor.AudioPreprocessor):
    """Audio processor performs min max scaling of an audio signal"""

    def __init__(self, max_val: int = 32768, min_val: int = -32767, percentile: float = 99.99):
        """
        Args:
            max_val: int
                The max value of an output signal (e.g. 32768 for 16bit audio signal).
            min_val: int
                The min value of an output signal (e.g. -32767 for 16bit audio signal).
            percentile: float in range [0, 1]
                Percentile to select current max value of a signal (to avoid outlier max values).
        """
        self._max_val = max_val
        self._min_val = min_val
        self._percentile = percentile

    def process(self, audio: np.ndarray) -> np.ndarray:
        k = self._max_val / np.percentile(np.abs(audio), self._percentile)
        return np.clip(audio * k, self._min_val, self._max_val)
