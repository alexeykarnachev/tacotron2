from abc import ABC, abstractmethod


class Vocoder(ABC):

    def __init__(self, *args, **kwargs):
        self.device = kwargs['device']
        self.args = args
        self.kwargs = kwargs

        self.model = None
        self.load_state()

    def to(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()

    @abstractmethod
    def load_state(self):
        pass

    @abstractmethod
    def infer(self, mel_outputs_postnet, *args, **kwargs):
        pass
