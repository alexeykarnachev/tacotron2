import logging
import os
from pathlib import Path
from typing import Tuple

import flasgger
import flask
import flask_basicauth

import rnd_utilities

from tacotron2.app.syntesis import defaults
from tacotron2.app.syntesis import views
from tacotron2.evaluators import BaseEvaluator
from tacotron2.evaluators.utils import get_evaluator, get_encoder_from_checkpoint
from tacotron2.hparams import HParams
from tacotron2.vocoder_interfaces._vocoder import Vocoder


def prepare_logging() -> logging.Logger:
    """Configures logging.

    Returns (logging.Logger):
        Logger object.
    """
    logger = logging.getLogger(__name__)
    log_dir = defaults.APP_DIR / 'logs'
    log_dir.mkdir(exist_ok=True, parents=False)

    return logger


def check_validness(s):
    if len(s) == 0:
        return False, 'Your string is empty!'
    elif len(s) > 150:
        return False, 'Your string is too long (please provide string less than 150 symbols)!'
    elif len(re.findall(r"[a-zA-Z]", s)) > 0:
        return False, 'Latin letters is in your string!'
    else:
        return True, 'Correct request'


def _add_app_routes(app, basic_auth, evaluator, wav_folder, logger):
    app.add_url_rule(
        '/speak/',
        view_func=basic_auth.required(
            views.Speak.as_view(
                'speak',
                evaluator=evaluator,
                wav_folder=wav_folder,
                logger=logger
            )
        ),
        methods=['POST']
    )

    app.add_url_rule(
        '/version',
        view_func=basic_auth.required(
            views.VersionView.as_view(
                'version'
            )
        ),
        methods=['GET']
    )

    app.add_url_rule(
        '/healthCheck',
        view_func=basic_auth.required(
            views.HealthCheckView.as_view(
                'healthCheck'
            )
        ),
        methods=['GET']
    )


def prepare_app(username: str, password: str) -> Tuple[flask.Flask, flask_basicauth.BasicAuth]:
    """Configures application.
    Sets the basic auth. credentials and specifies json encoder class.

    Returns (Tuple):
        Application and BasicAuth objects.
    """
    app = flask.Flask(defaults.APP_NAME)
    app.config['JSON_AS_ASCII'] = False
    app.json_encoder = rnd_utilities.CustomJsonEncoder
    app.config['BASIC_AUTH_USERNAME'] = username
    app.config['BASIC_AUTH_PASSWORD'] = password
    app.config['BASIC_AUTH_FORCE'] = True

    basic_auth = flask_basicauth.BasicAuth(app)

    return app, basic_auth


def prepare() -> flask.Flask:
    """Prepares application object.

    Returns (Flask):
        Flask application object.
    """
    logger = prepare_logging()
    config_file = os.getenv('APP_CONFIG')
    config = rnd_utilities.load_json(defaults.APP_DIR / config_file)

    wav_folder = defaults.APP_DIR / 'wavs'
    wav_folder.mkdir(exist_ok=True, parents=True)

    # evaluator = get_evaluator(
    #     evaluator_classname=config['evaluator_classname'],
    #     encoder_params=config['encoder_params'],
    #     vocoder_params=config['vocoder_params'],
    #     use_denoiser=config['use_denoiser'],
    #     device=config['device']
    # )
    encoder = get_encoder_from_checkpoint(ckpt_path=encoder_ckpt_path, device=device)
    vocoder = get_vocoder_from_checkpoint(ckpt_path=vocoder_ckpt_path, device=device)
    evaluator = BaseEvaluator(
        encoder=encoder,
        vocoder=vocoder,
    )

    # Flask application
    app, basic_auth = prepare_app(config['basic_auth_username'], config['basic_auth_password'])
    flasgger.Swagger(app)
    _add_app_routes(app, basic_auth, evaluator, wav_folder, logger)

    logger.info('All application objects have been initialized successfully.')

    return app
