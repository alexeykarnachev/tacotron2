import numpy as np

from tacotron2.callbacks._callback import Callback
from tacotron2.learner import Learner


class ReduceLROnPlateauCallback(Callback):
    """Callback which reduces learning rate on loss plateau"""

    def __init__(self, patience: int, reduce_factor: float):
        """
        :param patience: int, number of epochs without loss improvements before lr reduce
        :param reduce_factor: float, multiplier to apply to learning rate on reduction
        """
        self.patience = patience
        self.reduce_factor = reduce_factor
        self.min_loss = np.inf
        self.counter = 0
        self.prev_reduce_loss = -np.inf

    def on_epoch_end(self, learner: Learner):
        cur_loss = learner.eval(learner.valid_dl)

        if np.isfinite(cur_loss):
            if cur_loss < self.min_loss:
                self.min_loss = cur_loss
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= self.patience and cur_loss >= self.prev_reduce_loss:
                self.prev_reduce_loss = cur_loss
                for g in learner.optimizer.param_groups:
                    g['lr'] *= self.reduce_factor

                self.counter = 0
