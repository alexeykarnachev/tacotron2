import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, DistributedSampler

from tacotron2.factory import Factory
from tacotron2.hparams import HParams
from tacotron2.models._layers import TacotronSTFT
from tacotron2.utils import load_wav_to_torch, load_filepaths_and_text


class TextMelDataset(torch.utils.data.Dataset):
    """Texts and mel-spectrograms dataset
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
    """

    def __init__(self, meta_file_path: Path, tokenizer_class_name: str, load_mel_from_disk: bool, max_wav_value,
                 sampling_rate, filter_length, hop_length, win_length, n_mel_channels, mel_fmin, mel_fmax,
                 n_frames_per_step):
        """
        :param meta_file_path: Path, value separated text meta-file which has two fields:
            - relative (from the meta-file itself) path to the wav audio sample
            - audio sample text
            Fields must be separated by '|' symbol
        :param tokenizer_class_name: str, tokenizer class name. Must be importable from tacotron2.tokenizers module.
            If you have implemented custom tokenizer, add it's import to tacotron2.tokenizers.__init__.py file
        :param load_mel_from_disk:
        :param max_wav_value:
        :param sampling_rate:
        :param filter_length:
        :param hop_length:
        :param win_length:
        :param n_mel_channels:
        :param mel_fmin:
        :param mel_fmax:
        :param n_frames_per_step:
        """

        self.root_dir = meta_file_path.parent
        self.audiopaths_and_text = load_filepaths_and_text(meta_file_path)
        self.tokenizer = Factory.get_object(f'tacotron2.tokenizers.{tokenizer_class_name}')

        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.n_frames_per_step = n_frames_per_step
        self.load_mel_from_disk = load_mel_from_disk

        self.stft = TacotronSTFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            sampling_rate=sampling_rate,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax
        )

    @classmethod
    def from_hparams(cls, hparams: HParams, is_valid: bool):
        """Build class instance from hparams map
        If you create dataset instance via this method, make sure, that meta_train.txt (if is_valid==False) or
            meta_valid.txt (is is_valid==True) exists in the dataset directory
        :param hparams: HParams, dictionary with parameters
        :param is_valid: bool, get validation dataset or not (train)
        :return: TextMelLoader, dataset instance
        """
        param_names = inspect.getfullargspec(cls.__init__).args
        params = dict()
        for param_name in param_names:
            param_value = cls._get_param_value(param_name=param_name, hparams=hparams, is_valid=is_valid)

            if param_value is not None:
                params[param_name] = param_value

        obj = cls(**params)
        return obj

    @staticmethod
    def _get_param_value(param_name: str, hparams: HParams, is_valid: bool) -> Any:
        if param_name == 'self':
            value = None
        elif param_name == 'meta_file_path':
            data_directory = Path(hparams.data_directory)
            postfix = 'valid' if is_valid else 'train'
            value = data_directory / f'meta_{postfix}.txt'
            if not value.is_file():
                raise FileNotFoundError(f"Can't find {str(value)} file. Make sure, that file exists")
        else:
            value = hparams[param_name]

        return value

    def get_mel_text_pair(self, audiopath, text):
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return [text, mel]

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)

            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)

            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text: str):
        text_norm = torch.IntTensor(self.tokenizer.encode(text))
        return text_norm

    def __getitem__(self, index):
        file_path, text = self.audiopaths_and_text[index]
        file_path = self.root_dir / file_path
        item = self.get_mel_text_pair(file_path, text)
        return item

    def __len__(self):
        return len(self.audiopaths_and_text)

    @staticmethod
    def get_collate_function(pad_id, n_frames_per_step):
        collate = TextMelCollate(pad_id=pad_id, n_frames_per_step=n_frames_per_step)

        return collate

    def get_data_loader(self, batch_size: int, is_distributed: bool, shuffle: bool):
        """Construct DataLoader object from the Dataset object

        :param is_distributed: bool, set distributed sampler or not
        :param batch_size: int, batch size
        :param shuffle: bool, shuffle data or not
        :return: DataLoader
        """

        sampler = DistributedSampler(self, shuffle=shuffle) if is_distributed else None
        shuffle = shuffle if sampler is None else False

        collate_fn = self.get_collate_function(pad_id=self.tokenizer.pad_id, n_frames_per_step=self.n_frames_per_step)
        dataloader = DataLoader(
            self,
            num_workers=1,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn,
            shuffle=shuffle
        )

        return dataloader


class TextMelCollate:
    def __init__(self, pad_id, n_frames_per_step):
        self.pad_id = pad_id
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram

        :return: dict, batch dictionary with 'x', 'y' fields, each of which contains tuple of tensors
        """

        text, mel = list(zip(*batch))
        input_lengths, ids_sorted_decreasing = torch.sort(torch.tensor([len(x) for x in text]), descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(text), max_input_len)
        text_padded.fill_(self.pad_id)
        for i in range(len(ids_sorted_decreasing)):
            text_ = text[ids_sorted_decreasing[i]]
            text_padded[i, :text_.size(0)] = text_

        # Right zero-pad mel-spec
        num_mels = mel[0].size(0)
        max_target_len = max([x.size(1) for x in mel])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(mel), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(mel), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(mel))
        for i in range(len(ids_sorted_decreasing)):
            mel_ = mel[ids_sorted_decreasing[i]]
            mel_padded[i, :, :mel_.size(1)] = mel_
            gate_padded[i, mel_.size(1) - 1:] = 1
            output_lengths[i] = mel_.size(1)

        max_len = torch.max(input_lengths.data).item()

        batch_dict = {
            "x": [text_padded, input_lengths, mel_padded, max_len, output_lengths],
            "y": [mel_padded, gate_padded]
        }

        return batch_dict
