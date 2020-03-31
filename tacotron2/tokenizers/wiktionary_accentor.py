from typing import List
from pathlib import Path
import requests
import pymorphy2
from bs4 import BeautifulSoup
from rnd_utilities.file_utilities import load_json
import multiprocessing as mp
from tqdm import tqdm


_WIKTIONARY_URL = 'https://ru.wiktionary.org/wiki/Служебная:Поиск?search={}&go=Перейти'


class WiktionaryAccentor:
    def __init__(self, max_n_retries=2):
        self.max_n_retries = max_n_retries
        self.word2accents = load_json(Path(__file__).parent / 'data/accents_dict.json')
        self.morph_analyzer = pymorphy2.MorphAnalyzer()

    def get_accent(self, word: str, mode: str) -> str:
        if mode == 'offline':
            return self._get_accent_offline(word)
        elif mode == 'online':
            return self._get_accent_online(word)

    def _get_accent_offline(self, word: str) -> str:
        index = self.word2accents.get(word, None)
        if index:
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

    def parse_data(self, data: List[str], n_processes=4) -> List:
        with mp.Pool(n_processes) as pool:
            results = list(tqdm(pool.imap(self.get_from_wiki, data), total=len(data)))
            pool.close()
            pool.join()
        return results

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
