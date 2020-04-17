import datetime
import json
import os
import pickle
import random
from pathlib import Path
from typing import Union, Dict, Sequence, List

import numpy as np
import torch
import yaml

from tacotron2.factory import Factory
from tacotron2.json_encoder import CustomJSONEncoder


def load_yaml(path: Path):
    with open(str(path), 'r') as file:
        _yaml = yaml.full_load(file)

    return _yaml


def dump_yaml(_object: dict, path: Path):
    with open(str(path), 'w') as file:
        _yaml = yaml.dump(_object, file)

    return _yaml


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.LongTensor(range(0, max_len)).to(lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_filepaths_and_text(meta_file_path: Path, split="|"):
    meta_file_path = Path(meta_file_path)
    with meta_file_path.open(encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
        filepaths_and_text = [[str(x[0]), x[1]] for x in filepaths_and_text]

    return filepaths_and_text


def to_device_sequence(inp, device):
    if hasattr(inp, 'to'):
        inp = inp.to(device)
    else:
        try:
            for i in range(len(inp)):
                inp[i] = to_device(inp[i], device)
        except TypeError:
            pass

    return inp


def to_device_dict(inp: dict, device):
    return {k: to_device_sequence(v, device) for k, v in inp.items()}


def to_device(inp: Union[Dict, Sequence], device: Union[str, torch.device]) -> Union[Dict, Sequence]:
    """Sends input (each tensor from dict with tensors or sequence with tensors) to device.

    Args:
        inp: Dictionary or sequence with tensors.
        device: Device name.

    Returns:
        The same container as an input container, but with all tensors at specified device.
    """
    if isinstance(inp, Dict):
        return to_device_dict(inp, device)
    else:
        return to_device_sequence(inp, device)


def seed_everything(seed):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_cur_time_str():
    return datetime.datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S__%f')[:-3]


def load_json(file):
    with open(file) as f:
        return json.load(f)


def load_object(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def dump_object(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def dump_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, cls=CustomJSONEncoder, indent=2, ensure_ascii=False)


def prepare_dataloaders(hparams):
    model2dataset = {
        'Tacotron2': 'TextMelDataset',
        'Tacotron2Embedded': 'TextMelEmbeddingDataset'
    }

    dataset_class_name = model2dataset[hparams.model_class_name]
    dataset_class = Factory.get_class(f'tacotron2.datasets.{dataset_class_name}')

    dataloaders = []
    for flag in [False, True]:
        dataset = dataset_class.from_hparams(hparams, is_valid=flag)

        # TODO: get this sample embedding dimension automatically in the model???
        hparams.sample_embedding_dim = getattr(dataset, 'sample_embedding_dim', 0)

        dataloader = dataset.get_data_loader(hparams.batch_size)
        dataloaders.append(dataloader)

    hparams.n_symbols = len(dataloaders[0].dataset.tokenizer.id2token)

    return dataloaders
