import pathlib
import numpy as np

# Application settings
APP_DIR = pathlib.Path(__file__).parents[3] / 'app' / 'syntesis'
APP_NAME = 'TTS'

# Full text settings
RATE = 22050
SILENCE_DURATION = 0.3
SILENCE_PART = np.array([0] * int(SILENCE_DURATION * RATE))
MAX_SINGLE_LEN = 120

# View presentation and validation
MIN_UTTERANCE_LENGTH = 1
MAX_UTTERANCE_LENGTH = 1000
UTTERANCE = 'Привет! Я синтезированный голос.'