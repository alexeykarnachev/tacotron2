import marshmallow
from marshmallow import validate
from tacotron2.app.syntesis import defaults
from tacotron2.app.syntesis import fields


def _get_len_val(min_val, max_val):
    return validate.Length(min=min_val, max=max_val)


class SpeakRequestSchema(marshmallow.Schema):
    """Defines generator variable (which could be posted by client) parameters."""

    utterance = marshmallow.fields.Str(
        description="The higher the value is, the more diverse replies will be generated.",
        required=True, example=defaults.UTTERANCE,
        validate=_get_len_val(defaults.MIN_UTTERANCE_LENGTH, defaults.MAX_UTTERANCE_LENGTH)
    )


class SpeakResponseSchema(marshmallow.Schema):
    """Defines reply schema for the `say` route."""

    reply = fields.BytesField(
        description='Output bytes which can be converted to raw .wav file.',
        required=True
    )
