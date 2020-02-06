import librosa
import numpy as np

from tacotron2.audio_preprocessors._audio_preprocessor import AudioPreprocessor


class SilenceTrimmer(AudioPreprocessor):
    """Audio processor which trims silence parts on the beginning and on the end of input audio signal"""

    def __init__(self, db_threshold: int):
        """
        :param db_threshold: int, trimming threshold
        """
        self.db_threshold = db_threshold

    def process(self, audio: np.ndarray) -> np.ndarray:
        data, _ = librosa.effects.trim(
            audio.astype(np.float32), top_db=self.db_threshold)

        return data
