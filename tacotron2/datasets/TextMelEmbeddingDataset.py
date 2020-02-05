from pathlib import Path
from typing import Dict, Tuple, Any

import torch
import torch.utils.data
from torch.utils.data import DataLoader, DistributedSampler

from tacotron2.datasets.TextMelDataset import TextMelDataset
from tacotron2.hparams import HParams
from tacotron2.utils import load_object


class TextMelEmbeddingDataset(TextMelDataset):
    """Texts and mel-spectrograms dataset + samples embeddings
    1) loads audio, text pairs and samples embeddings
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
    """

    def __init__(self, meta_file_path: Path, embeddings_file_path: Path, tokenizer_class_name: str,
                 load_mel_from_disk: bool, max_wav_value, sampling_rate, filter_length, hop_length, win_length,
                 n_mel_channels, mel_fmin, mel_fmax, n_frames_per_step):
        """
        :param meta_file_path: Path, value separated text meta-file which has two fields:
            - relative (from the meta-file itself) path to the wav audio sample
            - audio sample text
            Fields must be separated by '|' symbol
        :param embeddings_file_path: Path, pickled file with samples embeddings. It presents a dictionary with 2 fields:
            - filename2id (a dictionary which maps specific wav file name to its embedding index)
            - embeddings (numpy 2d array of shape (N, D), where N is a number of embeddings, D is an embeddings dim.)
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

        super().__init__(meta_file_path, tokenizer_class_name, load_mel_from_disk, max_wav_value, sampling_rate,
                         filter_length, hop_length, win_length, n_mel_channels, mel_fmin, mel_fmax, n_frames_per_step)

        self.embeddings_dict = self._get_embeddings_dict()
        self.embeddings_dim = self.embeddings_dict['embeddings'].shape[1]

    def _get_embeddings_dict(self):
        embeddings_dict = load_object(self.embeddings_file_path)
        used_indexes = []
        used_file_paths = []

        for file_path, _ in self.audiopaths_and_text:
            try:
                index = embeddings_dict['filename2id'][str(file_path)]
            except KeyError:
                raise KeyError(f'Filename {str(file_path)} is not presented in the embeddings filename2id map')

            used_indexes.append(index)
            used_file_paths.append(file_path)

        embeddings_dict = {
            'filename2id': {str(file_path): i for i, file_path in enumerate(used_file_paths)},
            'embeddings': embeddings_dict['embeddings'][used_indexes]
        }

        return embeddings_dict

    @staticmethod
    def _get_param_value(param_name: str, hparams: HParams, is_valid: bool) -> Any:
        if param_name == 'embeddings_file_path':
            data_directory = Path(hparams.data_directory)
            value = data_directory / f'wav_embeddings.pkl'
        else:
            value = super()._get_param_value(param_name=param_name, hparams=hparams, is_valid=is_valid)

        return value

    def __getitem__(self, index):
        file_path, text = self.audiopaths_and_text[index]
        item = self.get_mel_text_pair((file_path, text))
        wav_embedding_index = self.embeddings_dict['filename2id'][str(file_path)]
        wav_embedding = self.embeddings_dict['embeddings'][wav_embedding_index]
        item.append(wav_embedding)
        return item

    def __len__(self):
        return len(self.audiopaths_and_text)

    @staticmethod
    def get_collate_function(pad_id, n_frames_per_step):
        def collate(batch) -> Dict[str, Tuple]:
            """Collate's training batch from normalized text and mel-spectrogram
            
            :return: dict, batch dictionary with 'x', 'y' fields, each of which contains tuple of tensors
            """
            # Right zero-pad all one-hot text sequences to max input length
            input_lengths, ids_sorted_decreasing = torch.sort(
                torch.LongTensor([len(x[0]) for x in batch]),
                dim=0, descending=True)
            max_input_len = input_lengths[0]

            text_padded = torch.LongTensor(len(batch), max_input_len)
            text_padded.fill_(pad_id)
            for i in range(len(ids_sorted_decreasing)):
                text = batch[ids_sorted_decreasing[i]][0]
                text_padded[i, :text.size(0)] = text

            # Right zero-pad mel-spec
            num_mels = batch[0][1].size(0)
            max_target_len = max([x[1].size(1) for x in batch])
            if max_target_len % n_frames_per_step != 0:
                max_target_len += n_frames_per_step - max_target_len % n_frames_per_step
                assert max_target_len % n_frames_per_step == 0

            # include mel padded and gate padded
            mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
            mel_padded.zero_()
            gate_padded = torch.FloatTensor(len(batch), max_target_len)
            gate_padded.zero_()
            output_lengths = torch.LongTensor(len(batch))
            for i in range(len(ids_sorted_decreasing)):
                mel = batch[ids_sorted_decreasing[i]][1]
                mel_padded[i, :, :mel.size(1)] = mel
                gate_padded[i, mel.size(1) - 1:] = 1
                output_lengths[i] = mel.size(1)

            max_len = torch.max(input_lengths.data).item()

            batch_dict = {
                "x": (text_padded, input_lengths, mel_padded, max_len, output_lengths),
                "y": (mel_padded, gate_padded)
            }

            return batch_dict

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
