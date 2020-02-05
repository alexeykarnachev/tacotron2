import argparse
import re
import pandas as pd
from tqdm import tqdm
from pandas.core.common import flatten

from russian_g2p.Transcription import Transcription


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='pd.Dataframe with train meta location')
parser.add_argument('-o', '--output', type=str, help='Where to store dict')
args = parser.parse_args()

word_regexp = re.compile(r'[А-яЁё]+')
transcriptor = Transcription()


if __name__ == '__main__':
    data = pd.read_csv(args.input, header=None, sep='|', names=['path', 'sentence', 'speaker'])
    sentences = data['sentence'].values

    phonemes_dict = {}
    for sent in tqdm(sentences):
        words_matches = list(word_regexp.finditer(sent))
        for i_word_match, word_match in enumerate(words_matches):
            matched_word = word_match.group(0)
            matched_word_tokens = phonemes_dict.get(matched_word, None)
            if matched_word_tokens is None:
                matched_word_tokens = flatten(transcriptor.transcribe([matched_word]))
                phonemes_dict[matched_word] = matched_word_tokens

    with open(args.output, 'w') as file:
        for k, v in phonemes_dict.items():
            file.write(f"{k} {' '.join([x for x in v])}" + "\n")
