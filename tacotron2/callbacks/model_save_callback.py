import re
from pathlib import Path

import numpy as np
import torch

from tacotron2.callbacks._callback import Callback
from tacotron2.learner import Learner


class ModelSaveCallback(Callback):
    def __init__(self, hold_n_models: int, models_dir: Path, ):
        """
        Callback which saves the model
        :param hold_n_models: int, how many models to store (old models will be overwritten by the new ones)
        :param models_dir: output models directory
        """
        self.hold_n_models = hold_n_models
        self.models_dir = Path(models_dir)
        self.best_val_loss = np.inf

        self.models_dir.mkdir(exist_ok=False, parents=True)

    def on_eval_end(self, learner: Learner):
        self._save(learner)
        self._save_best(learner)

    def on_train_end(self, learner):
        self._save(learner)

    def _save(self, learner: Learner):
        self._remove_old_models(models_dir=self.models_dir, hold_n_models=self.hold_n_models)
        self._save_model(learner, models_dir=self.models_dir)

    def _save_best(self, learner: Learner):
        current_val_loss = learner.valid_loss
        if np.isfinite(current_val_loss):
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self._save_model(learner, models_dir=self.models_dir, best=True)

    @staticmethod
    def _remove_old_models(models_dir: Path, hold_n_models: int):
        file_paths = [path for path in list(models_dir.iterdir()) if re.match(r'model_\d+.pth', path.name)]

        if len(file_paths) >= hold_n_models:
            model_steps = [int(re.findall(r'\d+', file_path.name)[0]) for file_path in file_paths]
            sorted_file_path_ids = np.argsort(model_steps)
            n_models_to_remove = (len(file_paths) - hold_n_models) + 1
            for file_path_id in sorted_file_path_ids[:n_models_to_remove]:
                file_path = file_paths[file_path_id]
                file_path.unlink()

    @staticmethod
    def _save_model(learner: Learner, models_dir: Path, best: bool = False):
        save_dict = {
            "model_state_dict": learner.model.state_dict(),
            "optimizer_state_dict": learner.optimizer.state_dict(),
            "overall_step": learner.overall_step,
            "n_epochs": learner.n_epochs,
            "cur_epoch": learner.cur_epoch,
            "n_epoch_steps": learner.n_epoch_steps,
            "valid_loss": learner.valid_loss,
            "train_loss": learner.train_loss
        }
        if not best:
            model_file = models_dir / f'model_{learner.overall_step}.pth'
        else:
            model_file = models_dir / 'model_best.pth'

        torch.save(save_dict, str(model_file))
