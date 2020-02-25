import os
import yaml
from pathlib import Path
import re

import librosa
from flask import Flask, request
from flask import send_file, make_response

from tacotron2.hparams import HParams
from tacotron2.evaluators import get_evaluator


with open(os.getenv('APP_CONFIG')) as cfg_file:
    CONFIG = yaml.load(cfg_file)

app_folder_path = Path(CONFIG['app_folder'])
app_folder_path.mkdir(exist_ok=True, parents=True)

hparams_tacotron = HParams.from_yaml(CONFIG['encoder_hparams_path'])
hparams_wg = HParams.from_yaml(CONFIG['vocoder_hparams_path'])
hparams_tacotron.n_symbols = 152

evaluator = get_evaluator(
    evaluator_classname=CONFIG['evaluator_classname'],
    encoder_hparams=hparams_tacotron,
    encoder_checkpoint_path=CONFIG['encoder_checkpoint_path'],
    vocoder_hparams=hparams_wg,
    vocoder_checkpoint_path=CONFIG['vocoder_checkpoint_path'],
    use_denoiser=CONFIG['use_denoiser'],
    device=CONFIG['device'])

# Flask application
app = Flask(__name__)

# Utilities
def check_validness(s):
    if len(s) == 0:
        return False, 'Your string is empty!'
    elif len(s) > 150:
        return False, 'Your string is too long (please provide string less than 150 symbols)!'
    elif len(re.findall(r"[a-zA-Z]", s)) > 0:
        return False, 'Latin letters is in your string!'
    else:
        return True, 'Correct request'

# Application endpoint
@app.route('/speak/', methods=['POST'])
def speak():
    if request.method == 'POST':
        context_text = request.json

        is_valid, reason = check_validness(context_text)

        if not is_valid:
            return make_response(reason, 400)

        tmp_path = str(app_folder_path / str(hash(context_text))) +  '.wav'
        audio, (_, _, _) = evaluator.synthesize(context_text)
        librosa.output.write_wav(str(tmp_path), audio.cpu().numpy().flatten(), 22050)

        return send_file(open(tmp_path, "rb"), mimetype="application/wav")
