import numpy as np
import librosa

from tacotron2.audio_preprocessors._audio_preprocessor import AudioPreprocessor


class SilenceTrimmer(AudioPreprocessor):
    def __init__(self, db_threshhold: int):
        self.db_threshhold = db_threshhold

    def process(self, audio):
        data, _ = librosa.effects.trim(
            audio.astype(np.float32), top_db=self.db_threshhold)

        return data