from abc import ABC, abstractmethod

import numpy as np


class AudioPreprocessor(ABC):
    """Abstract class for audio processors"""

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        return self.process(audio)

    @abstractmethod
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Abstract method which must be implemented in each processor children class

        :param audio: np.ndarray, input audio signal as a numpy 1d array
        :return: np.ndarray, processed audio signal
        """
        pass
