import argparse
import shutil
import warnings
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW

from tacotron2.callbacks.model_save_callback import ModelSaveCallback
from tacotron2.callbacks.reduce_lr_on_plateau_callback import ReduceLROnPlateauCallback
from tacotron2.callbacks.tensorboard_logger_callback import TensorBoardLoggerCallback
from tacotron2.factory import Factory
from tacotron2.hparams import HParams
from tacotron2.learner import Learner
from tacotron2.utils import seed_everything, get_cur_time_str, dump_json, prepare_dataloaders

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Run BERT based Bi-Encoder experiment')

parser.add_argument(
    '--experiments_dir', type=Path, required=True, help='Root directory of all your experiments'
)
parser.add_argument(
    '--hparams_file', type=Path, required=True, help='Path to the hparams yaml file'
)
parser.add_argument(
    '--tb_logdir', type=Path, required=True, help='Tensorboard logs directory'
)

args = parser.parse_args()

hparams = HParams.from_yaml(args.hparams_file)
experiments_dir = args.experiments_dir
experiment_id = get_cur_time_str()
tb_logdir = args.tb_logdir / experiment_id

experiment_dir: Path = experiments_dir / experiment_id
experiment_dir.mkdir(exist_ok=False, parents=True)
shutil.copy(str(args.hparams_file), str(experiment_dir / 'hparams.yaml'))
dump_json(args.__dict__, experiment_dir / 'arguments.json')
models_dir = experiment_dir / 'models'

if __name__ == '__main__':
    seed_everything(hparams.seed)
    dl = prepare_dataloaders(hparams)

    model = Factory.get_class(f'tacotron2.models.{hparams.model_class_name}')(hparams)
    optimizer = AdamW(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    summary_writer = SummaryWriter(log_dir=tb_logdir)

    callbacks = [
        TensorBoardLoggerCallback(
            summary_writer=summary_writer
        ),
        ModelSaveCallback(
            save_each_n_steps=hparams.iters_per_checkpoint,
            hold_n_models=3,
            models_dir=models_dir
        )
    ]

    if None not in (hparams.lr_reduce_patience, hparams.lr_reduce_factor):
        callbacks.append(
            ReduceLROnPlateauCallback(
                patience=hparams.lr_reduce_patience,
                reduce_factor=hparams.lr_reduce_factor
            )
        )

    learner = Learner(
        model=model,
        optimizer=optimizer,
        callbacks=callbacks
    ).fit(
        dl=dl,
        n_epochs=hparams.epochs,
        device=hparams.device,
        accum_steps=hparams.accum_steps,
        eval_steps=hparams.iters_per_checkpoint,
        use_all_gpu=hparams.use_all_gpu,
        fp16_opt_level=hparams.fp16_opt_level,
        max_grad_norm=hparams.grad_clip_thresh
    )
