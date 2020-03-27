from tacotron2.tokenizers.russian_phoneme_tokenizer import RussianPhonemeTokenizer


def test_word_in_wiki():
    cl = RussianPhonemeTokenizer()
    result = cl.encode('дезоксирибонуклеиновая')
    true_value = [2, 13, 40, 110, 4, 46, 79, 40, 75, 40, 8, 4, 62, 98, 46, 55, 41, 62, 4, 102, 4, 44, 4, 3]
    assert true_value == result


def test_word_in_phoneme_dict():
    cl = RussianPhonemeTokenizer()
    result = cl.encode('игрушка')
    true_value = [2, 40, 32, 74, 99, 81, 46, 4, 3]
    assert true_value == result


def test_word_in_accents_dict():
    cl = RussianPhonemeTokenizer()
    result = cl.encode('абрисах')
    true_value = [2, 5, 8, 75, 40, 78, 4, 49, 3]
    assert true_value == result

