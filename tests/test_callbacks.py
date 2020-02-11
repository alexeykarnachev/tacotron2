from typing import Sequence

from tacotron2.callbacks.reduce_lr_on_plateau_callback import ReduceLROnPlateauCallback


def test_reduce_on_plateau_callback():
    class Learner:
        class _Optimizer:
            param_groups = [{'lr': 1}]

        def __init__(self, strategy: Sequence[float]):
            self.valid_dl = None
            self.strategy = strategy
            self.optimizer = self._Optimizer()
            self.cur_step = -1

        def eval(self, valid_dl):
            self.cur_step += 1
            return self.strategy[self.cur_step]

    learner = Learner(strategy=[9, 8, 7, 6, 5, 6, 4, 3, 4, 5, 3, 2, 3, 3])
    clb = ReduceLROnPlateauCallback(patience=2, reduce_factor=0.5)

    for _ in learner.strategy:
        clb.on_epoch_end(learner=learner)
        print(learner.optimizer.param_groups)

test_reduce_on_plateau_callback()