from pathlib import Path
from typing import Any, List

import torch

from tacotron2.datasets.text_mel_dataset import TextMelDataset, TextMelCollate
from tacotron2.hparams import HParams
from tacotron2.utils import load_object
from tacotron2.audio_preprocessors._audio_preprocessor import AudioPreprocessor


class TextMelEmbeddingDataset(TextMelDataset):
    """Texts and mel-spectrograms dataset + samples embeddings
    1) loads audio, text pairs and samples embeddings
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
    """

    def __init__(self, meta_file_path: Path, embeddings_file_path: Path, tokenizer_class_name: str,
                 load_mel_from_disk: bool, max_wav_value, sampling_rate, filter_length, hop_length, win_length,
                 n_mel_channels, mel_fmin, mel_fmax, n_frames_per_step, audio_preprocessors: List[AudioPreprocessor]):
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
                         filter_length, hop_length, win_length, n_mel_channels, mel_fmin, mel_fmax, n_frames_per_step,
                         audio_preprocessors)

        self.embeddings_dict = self._get_embeddings_dict(embeddings_file_path)
        self.sample_embedding_dim = self.embeddings_dict['embeddings'].shape[1]

    def _get_embeddings_dict(self, embeddings_file_path):
        embeddings_dict = load_object(embeddings_file_path)
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
            value = TextMelDataset._get_param_value(param_name=param_name, hparams=hparams, is_valid=is_valid)

        return value

    def __getitem__(self, index):
        file_path, text = self.audiopaths_and_text[index]
        item = self.get_mel_text_pair(self.root_dir / file_path, text)
        wav_embedding_index = self.embeddings_dict['filename2id'][str(file_path)]
        wav_embedding = self.embeddings_dict['embeddings'][wav_embedding_index]
        item.append(wav_embedding)
        return item

    @staticmethod
    def get_collate_function(pad_id, n_frames_per_step):
        return TextMelEmbeddingsCollate(pad_id, n_frames_per_step)


class TextMelEmbeddingsCollate(TextMelCollate):
    def __init__(self, pad_id, n_frames_per_step):
        super().__init__(pad_id, n_frames_per_step)

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram

        :return: dict, batch dictionary with 'x', 'y' fields, each of which contains tuple of tensors
        """
        text, mel, emb = list(zip(*batch))
        batch_dict = super().__call__(zip(text, mel))
        emb = torch.FloatTensor(emb)
        batch_dict["x"].append(emb)
        return batch_dict
