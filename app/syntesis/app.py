import argparse
import yaml
from pathlib import Path

import librosa
from flask import Flask, request
from flask import send_file

from tacotron2.hparams import HParams
from tacotron2.evaluators import get_evaluator


parser = argparse.ArgumentParser(description='Starts a syntesis app')
parser.add_argument('--config_path', type=str, help='Path to config file')
args = parser.parse_args()

with open(args.config_path) as cfg_file:
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

# Application endpoint
@app.route('/speak/', methods=['POST'])
def speak():
    if request.method == 'POST':
        context_text = request.json
        tmp_path = str(app_folder_path / hash(context_text) + '.wav')
        audio, (_, _, _) = evaluator.synthesize(context_text)
        librosa.output.write_wav(str(tmp_path), audio.numpy().flatten(), 22050)

        return send_file(open(tmp_path, "rb"), mimetype="application/wav")
