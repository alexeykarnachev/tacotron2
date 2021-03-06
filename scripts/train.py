import argparse
import math
import os
import time
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from numpy import finfo
from tqdm import tqdm

from tacotron2.distributed import apply_gradient_allreduce
from tacotron2.factory import Factory
from tacotron2.hparams import HParams
from tacotron2.logger import Tacotron2Logger
from tacotron2.loss_function import Tacotron2Loss
from tacotron2.utils import seed_everything, to_device_dict, prepare_dataloaders

warnings.filterwarnings("ignore")


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model_class_name = hparams.model_class_name
    model_cls = Factory.get_class(f'tacotron2.models.{model_class_name}')
    model = model_cls(hparams).to(hparams.device)

    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, valid_dataloader, criterion, iteration, n_gpus, logger, distributed_run, rank, device):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():

        val_loss = 0.0
        for i, batch_dict in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            batch_dict = to_device_dict(batch_dict, device=device)
            y_pred = model(batch_dict)
            loss = criterion(y_pred, batch_dict['y'])
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(val_loss, model, batch_dict['y'], y_pred, iteration)

    return val_loss


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    seed_everything(hparams.seed)

    train_dataloader, valid_dataloader = prepare_dataloaders(hparams)
    hparams.n_symbols = len(train_dataloader.dataset.tokenizer.id2token)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_dataloader)))

    patience = 0
    val_losses = []
    is_overflow = False

    model.train()
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        tqdm_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, batch_dict in tqdm_bar:
            batch_dict = to_device_dict(batch_dict, device=hparams.device)
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            y_pred = model(batch_dict)
            loss = criterion(y_pred, batch_dict['y'])

            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                tqdm_bar.set_postfix_str(
                    "Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                        iteration, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)

            iteration += 1

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):

                val_loss = validate(model, valid_dataloader, criterion, iteration, n_gpus, logger,
                                    hparams.distributed_run, rank,
                                    hparams.device)
                val_losses.append(val_loss)

                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)
                if hparams.lr_reduce:
                    if val_losses[-hparams.lr_reduce['patience']:][0] < val_losses[-1]:
                        patience += 1
                    else:
                        patience = 0

                    if patience >= hparams.lr_reduce['patience']:
                        for g in optimizer.param_groups:
                            g['lr'] = g['lr'] / hparams.lr_reduce['divisor']
                        patience = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, help='Directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str, help='Directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None, required=False, help='Checkpoint path')
    parser.add_argument('--warm_start', action='store_true', help='Load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1, required=False, help='Number of gpus')
    parser.add_argument('--rank', type=int, default=0, required=False, help='Rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name', required=False, help='Distributed group name')
    parser.add_argument('--hparams_file', type=Path, required=False, help='Path to the hyper parameters yaml file')

    args = parser.parse_args()
    hparams = HParams.from_yaml(args.hparams_file)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
