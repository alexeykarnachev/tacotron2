import http
from logging import Logger
from pathlib import Path

import flask
import librosa
import marshmallow
from flasgger import SwaggerView

from tacotron2.app.syntesis import schemas
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
            "description": "OK.",
            "schema": schemas.SpeakResponseSchema
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

        utterance = data['utterance']
        self.logger.info(f'Got utterance: {utterance}')

        tmp_path = str(self.wav_folder / str(hash(utterance))) + '.wav'
        audio, (_, _, _) = self.evaluator.synthesize(utterance)
        librosa.output.write_wav(tmp_path, audio.cpu().numpy().flatten(), 22050)
        self.logger.info(f'Wav stored at: {tmp_path}')

        with open(tmp_path, 'rb') as file:
            result = file.read()

        reply = {
            'reply': result
        }
        reply = schemas.SpeakResponseSchema().dump(reply)
        self.logger.info('Result was constructed.')

        return reply, http.HTTPStatus.OK
