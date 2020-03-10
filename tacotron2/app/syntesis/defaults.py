import pathlib

APP_DIR = pathlib.Path(__file__).parents[3] / 'app' / 'syntesis'
APP_NAME = 'TTS'

MIN_UTTERANCE_LENGTH = 1
MAX_UTTERANCE_LENGTH = 150
UTTERANCE = 'Привет! Я синтезированный голос.'