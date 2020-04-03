from pathlib import Path
from rnd_utilities.file_utilities import load_json
import gdown


_ACCENTS_FILE_ID = '1AjWop6_-ZErmbklE83U5pXpjTYgSM3vr'
_GOOGLE_DRIVE_PATH = "https://drive.google.com/uc?id={}".format(_ACCENTS_FILE_ID)


class WiktionaryAccentor:
    def __init__(self):
        accents_path = Path(__file__).parent / 'data/accents.json'
        if not accents_path.is_file():
            gdown.download(_GOOGLE_DRIVE_PATH, str(accents_path.absolute()))
        self.word2accents = load_json(accents_path)

    def get_accent(self, word: str) -> str:
        indexes = self.word2accents.get(word, None)
        if indexes:
            for index in indexes:
                word = word[:index] + '+' + word[index:]
            return word
        return word

