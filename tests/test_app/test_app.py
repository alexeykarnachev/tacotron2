import io
import json

import librosa
import numpy as np
import razdel
from requests import auth

import tacotron2
from tacotron2.app.syntesis.defaults import SILENCE_PART

_HEADERS = {
    "Authorization": auth._basic_auth_str('user', 'password'),
    "content-type": "application/json"
}

_VERSION = tacotron2.__version__
_UTTERANCES = [
    "Привет!",
    "Привет! Меня зовут Егор.",
    "Раз! Два! Три!"
]


def _get_post_speak_data(utterance, denoising_strength=0.05):
    data = {
        "utterance": utterance,
        "denoiser_strength": denoising_strength,
    }
    data = json.dumps(data)
    return data


def test_app(client):
    version = client.get("/version", headers=_HEADERS)
    assert json.loads(version.data) == _VERSION
    assert version.status_code == 200

    health_check = client.get("/healthCheck", headers=_HEADERS)
    assert health_check.data.decode() == 'Ok'
    assert version.status_code == 200

    for _utterance in _UTTERANCES:
        _denoising_strength = 0.05
        reply = client.post(
            "/speak/", headers=_HEADERS, data=_get_post_speak_data(_utterance, _denoising_strength)
        )

        io_stream = io.BytesIO(reply.data)
        audio = librosa.load(io_stream)[0]

        assert isinstance(audio, np.ndarray)
        assert len(audio) == len(_utterance)