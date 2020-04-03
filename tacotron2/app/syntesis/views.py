import http
from logging import Logger
from pathlib import Path

import flask
import librosa
import marshmallow
import numpy as np
from flasgger import SwaggerView
from razdel import sentenize

from tacotron2.app.syntesis import schemas
from tacotron2.app.syntesis import defaults
from tacotron2.evaluators import BaseEvaluator


class Speak(SwaggerView):
    parameters = [
        {
            "name": "utterance",
            "in": "body",
            "schema": schemas.SpeakRequestSchema,
            "required": True,
            "description": "POST body of the speak-endpoint."
        }
    ]
    responses = {
        200: {
            "description": "OK."
        },
        400: {
            "description": "Bad request.",
            "type": "string"
        }
    }

    def __init__(self, evaluator: BaseEvaluator, wav_folder: Path, logger: Logger):
        self.evaluator = evaluator
        self.wav_folder = wav_folder
        self.logger = logger

    def post(self):

        try:
            data = schemas.SpeakRequestSchema().load(flask.request.json)
        except marshmallow.exceptions.ValidationError as e:
            reply = flask.jsonify(str(e))
            return reply, http.HTTPStatus.BAD_REQUEST

        # get text from request:
        utterance = data['utterance']
        tmp_path = str(self.wav_folder / str(hash(utterance))) + '.wav'
        self.logger.info(f'Got utterance: {utterance}')

        # Construct 1 or more sentences for synthesis:
        if len(utterance) < defaults.MAX_SINGLE_LEN:
            sentences = [utterance]
        else:
            sentences = list(sentenize(utterance))

        # Iteratively synthesize each one:
        audio_parts = []
        for sentence in sentences:
            audio, (_, _, _) = self.evaluator.synthesize(sentence)
            audio_parts.append(audio.cpu().numpy().flatten())

        # Join all together with small-duration silence between sentences.
        full_audio = [
            np.concatenate([x, defaults.SILENCE_PART]) if i + 1 < len(audio_parts) else x
            for i, x in enumerate(audio_parts)
        ]
        full_audio = np.concatenate(full_audio)

        # Write wav and construct a base64 reply.
        librosa.output.write_wav(tmp_path, full_audio, defaults.RATE)
        self.logger.info(f'Wav stored at: {tmp_path}')

        with open(tmp_path, 'rb') as file:
            result = file.read()

        self.logger.info('Result was constructed.')

        return result, http.HTTPStatus.OK
