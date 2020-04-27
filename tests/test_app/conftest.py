import os
import unittest.mock

import numpy as np
import pytest
import rnd_utilities
import yaml

import torch

import tacotron2
from tacotron2.app.syntesis import utilities, defaults


class DummyEvaluator(object):
    def __init__(self):
        pass

    def synthesize(self, text, denoiser_strength=0.05):
        audio = torch.tensor(np.random.randn(len(text)))
        return audio, (audio, 0, 0)


@pytest.fixture
def client(tmp_path, monkeypatch):

    monkeypatch.setattr("tacotron2.app.syntesis.defaults.APP_DIR", tmp_path)
    tacotron2.app.syntesis.utilities.get_evaluator = unittest.mock.MagicMock()
    tacotron2.app.syntesis.utilities.get_evaluator.return_value = DummyEvaluator()

    config_path = tmp_path / 'app_config.json'

    hparams_encoder_path = tmp_path / "encoder_hparams_path.yaml"
    with hparams_encoder_path.open('w') as file:
        yaml.dump({"dummy": "dummy"}, file)

    hparams_vocoder_path = tmp_path / "vocoder_hparams_path.yaml"
    with hparams_vocoder_path.open('w') as file:
        yaml.dump({"dummy": "dummy"}, file)

    config = {
          "app_folder": str(tmp_path),
          "evaluator_classname": "BaseEvaluator",
          "encoder_hparams_path": str(hparams_encoder_path),
          "encoder_checkpoint_path": str(tmp_path / "encoder_checkpoint_path.pth"),
          "vocoder_hparams_path": str(hparams_vocoder_path),
          "vocoder_checkpoint_path": str(tmp_path / "vocoder_checkpoint_path.pth"),
          "use_denoiser": True,
          "device": "cpu",
          "basic_auth_username": "user",
          "basic_auth_password": "password"
    }

    rnd_utilities.dump_json(config, config_path)
    os.environ["APP_CONFIG"] = str(config_path)

    app = utilities.prepare()

    with app.test_client() as client:
        yield client
