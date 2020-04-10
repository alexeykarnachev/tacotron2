import marshmallow
from marshmallow import validate
from tacotron2.app.syntesis import defaults


def _get_len_val(min, max):
    return validate.Length(min=min, max=max)


class SpeakRequestSchema(marshmallow.Schema):
    """Defines generator variable (which could be posted by client) parameters."""

    utterance = marshmallow.fields.Str(
        description="Utterance to be speaked.",
        required=True, example=defaults.UTTERANCE,
        validate=_get_len_val(min=defaults.MIN_UTTERANCE_LENGTH, max=defaults.MAX_UTTERANCE_LENGTH)
    )
    denoiser_strength = marshmallow.fields.Float(
        description="Strength for denoising filter at postprocessing stage.",
        required=False, example=defaults.DENOISING_STRENGTH,
        validate=validate.Range(min=defaults.DENOISING_MIN, max=defaults.DENOISING_MAX)
    )
