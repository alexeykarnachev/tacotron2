import re
import requests
from pathlib import Path
from string import punctuation
from typing import List, Dict

from russian_g2p.Grapheme2Phoneme import Grapheme2Phoneme
from russian_g2p.modes.Phonetics import Phonetics

from tacotron2.tokenizers._tokenizer import Tokenizer
from tacotron2.tokenizers._utilities import replace_numbers_with_text, clean_spaces
from rnd_utilities.file_utilities import load_json

import pymorphy2
from bs4 import BeautifulSoup


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
        self.word2accents = load_json(Path(__file__).parent / 'data/accents_dict.json')
        self.word_regexp = re.compile(r'[А-яЁё]+')
        self.punctuation_regexp = re.compile(f'[{punctuation}]+')
        self.transcriptor = Grapheme2Phoneme()
        self.morph_analyzer = pymorphy2.MorphAnalyzer()

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

    def encode(self, text: str) -> List[int]:
        """Tokenize and encode text on phonemes-lvl

        :param text: str, input text
        :return: list, of phonemes ids
        """
        text = text.lower()
        text = replace_numbers_with_text(text, lang='ru')
        text = clean_spaces(text)

        token_ids = self._encode(text)
        # Hypotesis: bos\eos make training worse.
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

            # check in words2phonemes dict
            matched_word_tokens = self.word2phonemes.get(matched_word, None)
            if matched_word_tokens is None:
                # check in accents dict
                accented_word_index = self.word2accents.get(matched_word, None)

                if accented_word_index is None:
                    # try to find in wiki
                    wiki_accented_word = self.get_accent_from_wiki(matched_word)
                    if wiki_accented_word is not None:
                        # found in wiki!
                        matched_word = wiki_accented_word
                else:
                    # found in accents dict
                    matched_word = matched_word[:accented_word_index] + '+' + matched_word[accented_word_index:]

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

    def get_accent_from_wiki(self, word: str) -> str:

        # if we get word in case that is not nomn,
        # get wiki-page with word in nonm then search for original form mentioned on page
        parsed = self.morph_analyzer.parse(word)
        if parsed[0].tag.case != 'nomn':
            word = parsed[0].normal_form

        url = 'https://ru.wiktionary.org/wiki/Служебная:Поиск?search={}&go=Перейти'
        req = requests.get(url.format(word), allow_redirects=True, headers={'Content-Type': 'text/html; charset=UTF-8'})
        return self.parse_wiki_page(req.text, word)

    @staticmethod
    def parse_wiki_page(content: str, word: str) -> str:

        def delete_misc_symbols(s, hyphen=False):
            s = ''.join([x for x in s if ord(x) != 183])

            # delete grave accent
            s = s.replace('ѐ', 'е')
            s = s.replace('ѝ', 'и')
            if hyphen:
                return ''.join([x for x in s if ord(x) != 769 and ord(x) != 768])
            return s

        # parse wiki page
        soup = BeautifulSoup(content)
        if 'не существует' in soup.find("b").text:
            return None

        hypothesis = soup.find("b").text.replace('-', '').lower()

        # parse accented forms from wiki (since we parse page where word was in nomn case)
        tags = {delete_misc_symbols(x.text.lower().strip(), True): delete_misc_symbols(x.text.lower().strip()) for x in soup.find_all('td')}

        # check for е/ё
        res = [x for x in tags.keys() if x.replace('ё', 'е') == word]
        if len(res):
            hypothesis = tags[res[0]]

        # change ` to +
        accented = ['+' if ord(ch) == 769 else ch for ch in hypothesis]
        return ''.join(accented)
