import argparse
import copy
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import loggers
from rnd_utilities.datetime_utilities import get_cur_time_str

from tacotron2.utils import dump_yaml
from tacotron2.hparams import HParams, serialize_hparams
from tacotron2.pl_module import TacotronModule


def parse_args():
    parser = argparse.ArgumentParser(description='Run Tacotron experiment')
    parser.add_argument(
        '--hparams_file', type=Path, required=True,
        help='Path to the hparams yaml file'
    )
    parser.add_argument(
        '--experiments_dir', type=Path, required=True,
        help='Root directory of all your experiments'
    )
    parser.add_argument(
        '--experiment_name', type=str, required=False, default=None,
        help='Name of current experiment'
    )
    args = parser.parse_args()
    return args


# Now we dont have test of this train script, but when we get:
# TODO: replace with similar function from rnd_utilities.
def prepare_experiment(args) -> Path:

    experiments_dir: Path = args.experiments_dir
    experiments_dir.mkdir(exist_ok=True, parents=True)

    if args.experiment_name is None:
        experiment_name = get_cur_time_str()
    else:
        experiment_name: str = args.experiment_name
        if (experiments_dir / experiment_name).is_dir():
            experiment_name = experiment_name + '_' + get_cur_time_str()

    experiment_dir: Path = experiments_dir / experiment_name
    experiment_dir.mkdir(exist_ok=False)

    models_dir = experiment_dir / 'models'
    models_dir.mkdir(exist_ok=True)

    return experiment_dir


def get_trainer(_args: argparse.Namespace, _hparams: HParams) -> pl.Trainer:

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=_hparams.models_dir,
        verbose=True,
        save_top_k=_hparams.save_top_k
    )

    tb_logger_callback = loggers.TensorBoardLogger(
        save_dir=_hparams.tb_logdir, name='logs'

    )

    trainer_args = copy.deepcopy(_args.__dict__)
    trainer_args.update(
        {
            'logger': tb_logger_callback,
            'checkpoint_callback': model_checkpoint_callback,
            'max_epochs': _hparams.epochs,
            'gpus': _hparams.gpus,
            'val_check_interval ': _hparams.iters_per_checkpoint,
            'amp_level': _hparams.fp16_opt_level,
            'precision': _hparams.precision,
            'gradient_clip_val': _hparams.grad_clip_thresh,
            'accumulate_grad_batches': _hparams.accum_steps,
            'show_progress_bar': True,
            'progress_bar_refresh_rate': 1,
            'log_save_interval': 1,
        }
    )
    _trainer = pl.Trainer(**trainer_args)
    return _trainer


def main():
    args = parse_args()
    experiment_dir = prepare_experiment(args)

    hparams = HParams.from_yaml(args.hparams_file)
    dump_yaml(
        serialize_hparams(hparams.__dict__),
        experiment_dir / 'hparams.yaml')
    hparams['models_dir'] = experiment_dir / 'models'
    hparams['tb_logdir'] = experiment_dir

    module = TacotronModule(hparams)
    trainer = get_trainer(_args=args, _hparams=hparams)
    trainer.fit(module)


if __name__ == '__main__':
    main()