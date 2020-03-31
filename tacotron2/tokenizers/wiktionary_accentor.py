from typing import List
from pathlib import Path
import requests
import pymorphy2
from bs4 import BeautifulSoup
from rnd_utilities.file_utilities import load_json, dump_json
import multiprocessing as mp
from tqdm import tqdm


_WIKTIONARY_URL = 'https://ru.wiktionary.org/wiki/Служебная:Поиск?search={}&go=Перейти'


class WiktionaryAccentor:
    def __init__(self, do_lookup_in_wiki: bool, max_n_retries=2):
        self.do_lookup_in_wiki = do_lookup_in_wiki
        self.max_n_retries = max_n_retries
        self.word2accents = load_json(Path(__file__).parent / 'data/accents.json')
        self.morph_analyzer = pymorphy2.MorphAnalyzer()

    def get_accent(self, word: str) -> str:
        res = self._get_accent_offline(word)
        if res == word and self.do_lookup_in_wiki:
            return self._get_accent_online(word)
        return res

    def _get_accent_offline(self, word: str) -> str:
        indexes = self.word2accents.get(word, None)
        if indexes:
            for index in indexes:
                word = word[:index] + '+' + word[index:]
            return word
        return word

    def _get_accent_online(self, word: str) -> str:
        parsed = self.morph_analyzer.parse(word)
        if parsed[0].tag.case != 'nomn':
            word = parsed[0].normal_form

        variants = self.get_from_wiki(word)
        not_stressed_variants = [self.delete_stress_sign(v) for v in variants]
        if word in not_stressed_variants:
            index = not_stressed_variants.index(word)
            word = self.stress_sign_to_plus(variants[index])
        return self.delete_misc_symbols(word)

    def parse_data(self, data: List[str], output_name: Path, n_processes=4) -> None:
        def get_indexes_of_stress(s):
            return [i for i in range(len(s)) if ord(s[i]) == 769]

        with mp.Pool(n_processes) as pool:
            results = list(tqdm(pool.imap(self.get_from_wiki, data), total=len(data)))
            pool.close()
            pool.join()

        word_forms = []
        for res in results:
            for word in res:
                word_forms.extend(word.split())
        word_forms = list(set(word_forms))
        word_forms = [x.lower().strip().replace('*', '').replace('·', '') for x in word_forms]
        accents_dict = {self.delete_stress_sign(w): get_indexes_of_stress(w) for w in word_forms}
        dump_json(accents_dict, output_name)

    @staticmethod
    def get_from_wiki(word, max_n_tries=2) -> List:
        session = requests.Session()
        session.mount("https://", requests.adapters.HTTPAdapter(max_retries=max_n_tries))
        headers = {'Content-Type': 'text/html; charset=UTF-8'}
        try:
            req = session.get(_WIKTIONARY_URL.format(word), allow_redirects=True, headers=headers)
        except ConnectionError:
            print('Connection closed')
            return []
        except Exception as e:
            print('Unexpected error {}'.format(e))
            return []

        if req.status_code == 200:
            soup = BeautifulSoup(req.text, 'lxml')
            try:
                if 'не существует' in soup.find("b").text:
                    return []
                else:
                    variants = []
                    for x in soup.find_all('td', {'bgcolor': '#ffffff'}):
                        if x.br:
                            x = BeautifulSoup(str(x).replace('<br/>', ' '))
                        text = x.text.strip()
                        variants.append(text)
                    if not variants:
                        variants = [soup.find("b").text.replace('-', '').lower()]
                    return variants
            except AttributeError as e:
                print(e)
                return []
        else:
            return []

    @staticmethod
    def delete_misc_symbols(s: str) -> str:
        s = ''.join([x for x in s if ord(x) != 183])
        s = s.replace('ѐ', 'е').replace('ѝ', 'и').replace('а̀', 'а')  # delete grave accent
        return s

    def delete_stress_sign(self, word: str) -> str:
        word = ''.join([x for x in self.delete_misc_symbols(word) if ord(x) != 769 and ord(x) != 768])
        return word

    @staticmethod
    def stress_sign_to_plus(word: str) -> str:
        return ''.join(['+' if ord(ch) == 769 else ch for ch in word])
