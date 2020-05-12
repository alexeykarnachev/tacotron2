from abc import ABC, abstractmethod

import torch


class Vocoder(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def infer(self, encoder_output, **kwargs):
        pass
