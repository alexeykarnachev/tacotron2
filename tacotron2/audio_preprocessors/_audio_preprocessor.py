from abc import ABC, abstractmethod


class AudioPreprocessor(ABC):

    def __call__(self, audio):
        return self.process(audio)

    @abstractmethod
    def process(self, audio):
        pass