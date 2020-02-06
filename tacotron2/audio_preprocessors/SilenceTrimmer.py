import numpy as np
import librosa


class SilenceTrimmer(object):
    def __init__(self, db_threshhold: int):
        self.db_threshhold = db_threshhold

    def __call__(self, audio):
        data, _ = librosa.effects.trim(
            audio.astype(np.float32), top_db=self.db_threshhold)

        return data