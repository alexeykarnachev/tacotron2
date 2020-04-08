import re
from pathlib import Path
from string import punctuation
from typing import List, Dict, Union

from russian_g2p.Grapheme2Phoneme import Grapheme2Phoneme
from russian_g2p.modes.Phonetics import Phonetics

from tacotron2.tokenizers._tokenizer import Tokenizer
from tacotron2.tokenizers._utilities import replace_numbers_with_text, clean_spaces

from rnd_utilities import load_json


class RussianPhonemeTokenizer(Tokenizer):
    """Russian phonemes-lvl tokenizer
    It uses pre-calculated phonemes dictionary. If some specific word is not in the dictionary, then the
    russian_g2p.Transcription will be applied (https://github.com/nsu-ai/russian_g2p)
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

    @property
    def space_id(self):
        return self.token2id[' ']

    def __init__(self):
        self.russian_phonemes = sorted(Phonetics().russian_phonemes_set)
        self.pad = '<PAD>'
        self.unk = '<UNK>'
        self.bos = '<BOS>'
        self.eos = '<EOS>'
        self.id2token = [self.pad, self.unk, self.bos, self.eos] + self.russian_phonemes + list(punctuation) + [' ']
        self.token2id = {token: id_ for id_, token in enumerate(self.id2token)}
        assert len(self.id2token) == len(self.token2id)

        self.word2phonemes = self._read_phonemes_corpus(Path(__file__).parent / 'data/russian_phonemes_corpus.txt')

        try:
            accents_file_path = Path(__file__).parent / 'data/accents.json'
            self.word2accents = load_json(accents_file_path)
        except:
            raise FileNotFoundError(f'Accents dictionary can not be found at {accents_file_path}')

        self.word_regexp = re.compile(r'[А-яЁё]+')
        # Do we really need this?
        self.punctuation_regexp = re.compile(f'[{punctuation}]+')

        self.transcriptor = Grapheme2Phoneme()

    @staticmethod
    def _read_phonemes_corpus(file_path: Path) -> Dict[str, List[str]]:
        """Read pre-calculated phonemes corpus (word to phonemes list map)

        :param file_path: Path, path to the corpus file
        :return: dict, word to phonemes dictionary
        """
        phonemes_corpus = dict()
        with file_path.open() as f:
            for line in f.readlines():
                line_split = line.strip().split()
                phonemes_corpus[line_split[0].lower()] = line_split[1:]

        return phonemes_corpus

    def encode(self, text: Union[str, List[str]]) -> List[int]:
        """Tokenize and encode text on phonemes-lvl

        :param text: str, input text
        :return: list, of phonemes ids
        """
        if isinstance(text, str):
            text = text.lower()
            text = replace_numbers_with_text(text, lang='ru')
            text = clean_spaces(text)

            token_ids = self._encode(text)
        elif isinstance(text, list):
            token_ids = [self.token2id.get(x, self.unk_id) for x in text]
        else:
            raise AttributeError('Given text must be string or list of phonemes.')

        token_ids.insert(0, self.bos_id)
        token_ids.append(self.eos_id)

        return token_ids

    def _encode(self, text: str) -> List[int]:
        """Tokenize text on phonemes. Uses dictionary if word is presented, or calculate phonemes in runntime

        :param text: str, input text
        :return: list, of token ids
        """
        word_ids_sequences = []

        word_matches = list(self.word_regexp.finditer(text))
        for i_word_match, word_match in enumerate(word_matches):
            matched_word = word_match.group(0)

            matched_word_tokens = self.word2phonemes.get(matched_word, None)
            if matched_word_tokens is None:
                matched_word = self.get_accent(matched_word)
                matched_word_tokens = self.transcriptor.word_to_phonemes(matched_word)

            try:
                matched_word_ids = [self.token2id[token] for token in matched_word_tokens]
            except KeyError:
                raise KeyError(f'Some of word phonemes are not in the tokenizer tokens '
                               f'map (Word: {matched_word}, Tokens: {matched_word_tokens})')

            word_ids_sequences.append(matched_word_ids)

        not_word_substrings_split = self.word_regexp.split(text)

        all_ids = []

        for i, not_word_substring in enumerate(not_word_substrings_split):
            matched_not_word_ids = [self.token2id.get(token, self.unk_id) for token in not_word_substring]
            all_ids.extend(matched_not_word_ids)

            if i < len(not_word_substrings_split) - 1:
                all_ids.extend(word_ids_sequences[i])
        return all_ids

    def get_accent(self, word):
        indexes = self.word2accents.get(word, None)
        if indexes:
            for index in indexes:
                word = word[:index] + '+' + word[index:]
        return word
