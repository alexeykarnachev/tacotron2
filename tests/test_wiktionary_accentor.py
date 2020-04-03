import pytest


words = [
    'макромолекулы',
    'кислота',
    'безлимитищенский',
    'бело-лазорево-алые',
    'антиимпериалистически-демократическою',
    'бренд-менеджеру']
stressed_words = [
    'макромоле+кулы',
    'кислота+',
    'безлимитищенский',
    'бе+ло-лазо+рево-а+лые',
    'а+нтиимпериалисти+чески-демократи+ческою',
    'бре+нд-ме+неджеру']

test_data = [(words[i], stressed_words[i]) for i in range(len(words))]


@pytest.fixture
def wiktionary_accentor():
    from tacotron2.tokenizers.russian_phoneme_tokenizer import WiktionaryAccentor
    return WiktionaryAccentor()


@pytest.mark.parametrize("test_input,expected", test_data)
def test_stress(wiktionary_accentor, test_input, expected):
    result = wiktionary_accentor.get_accent(test_input)
    assert expected == result
