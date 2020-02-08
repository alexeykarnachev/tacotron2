from abc import ABC, abstractmethod
from string import punctuation
from typing import List

from tacotron2.tokenizers._tokenizer import Tokenizer


class GraphemeTokenizer(Tokenizer, ABC):
    """Abstract graphemes-lvl tokenizer
    Tokenizer which encodes text string into a sequence of grapheme indexes. It also allows to replace
    numbers with their word-representations
    """

    @property
    def pad_id(self):
        return 0

    @property
    def unk_id(self):
        return 1

    @property
    def bos_id(self):
        return 2

    @property
    def eos_id(self):
        return 3

    def __init__(self, language: str, letters: str):
        """
        :param language: str, tokenizer language
        :param letters: str, alphabet letters concatenated in one string
        """
        self.letters = letters
        self.alphabet = self.letters + punctuation + ' '
        self.pad = '<PAD>'
        self.unk = '<UNK>'
        self.bos = '<BOS>'
        self.eos = '<EOS>'
        self.id2token = [self.pad, self.unk, self.bos, self.eos] + list(self.alphabet)
        self.token2id = {token: id_ for id_, token in enumerate(self.id2token)}
        self.language = language

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token (graphemes) indexes

        :param text: str, input text
        :return: list, grapheme indexes
        """

        text = self._clean_text(text)
        token_ids = self._numericalize(text)

        return token_ids

    def _numericalize(self, text: str) -> List[int]:
        token_ids = [self.token2id.get(token, self.unk_id) for token in text]
        token_ids.insert(0, self.bos_id)
        token_ids.append(self.eos_id)
        return token_ids

    @abstractmethod
    def _clean_text(self, text: str) -> str:
        """Arbitrary text cleaning pipeline which must be implemented in the children classes

        :param text: str, input text
        :return: str, cleaned text
        """
        return text
