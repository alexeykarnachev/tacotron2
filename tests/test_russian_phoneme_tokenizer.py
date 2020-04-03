import pytest


words = ['дезоксирибонуклеиновая',
         'игрушка',
         'абрисах',
         'бренд-менеджеру']
tokens = [[2, 13, 40, 110, 4, 46, 79, 40, 75, 40, 8, 4, 62, 98, 46, 55, 41, 62, 4, 102, 4, 44, 4, 3],
          [2, 40, 32, 74, 99, 81, 46, 4, 3],
          [2, 5, 8, 75, 40, 78, 4, 49, 3],
          [2, 8, 75, 40, 62, 86, 131, 59, 25, 63, 40, 18, 106, 74, 98, 3]]
test_data = [(words[i], tokens[i]) for i in range(len(words))]


@pytest.fixture
def russian_phoneme_tokenizer():
    from tacotron2.tokenizers.russian_phoneme_tokenizer import RussianPhonemeTokenizer
    return RussianPhonemeTokenizer()


@pytest.mark.parametrize("test_input,expected", test_data)
def test_word_in_wiki(russian_phoneme_tokenizer, test_input, expected):
    result = russian_phoneme_tokenizer.encode(test_input)
    assert expected == result
