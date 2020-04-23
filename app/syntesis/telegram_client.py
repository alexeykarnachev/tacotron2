import http
import io
import argparse
import json
import os
import pathlib
from collections import defaultdict
from typing import Optional, Tuple

import librosa
import rnd_utilities
import aiohttp

from pydub import AudioSegment
from aiogram import Bot, Dispatcher, types, executor
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from tacotron2.app.syntesis import defaults


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str, required=True, help='Telegram bot configuration'
)
args = parser.parse_args()

config = rnd_utilities.load_json(pathlib.Path(args.config))

API_TOKEN = config['api_token']
AUTH = aiohttp.BasicAuth(
    login=config['user'],
    password=config['password']
)
HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

BOT = Bot(token=API_TOKEN)
BOT_FOLDER = config['bot_folder']
DP = Dispatcher(BOT)

VOICES = config['voices']
DEFAULT_VOICE = '__default__'
START_VOICE = config['default_voice']
USER_VOICES = defaultdict(lambda: START_VOICE)


def get_mp3_path(wav_basestring, text, sampling_rate):
    hashname = str(hash(text))
    io_stream = io.BytesIO(wav_basestring)
    audio = librosa.load(io_stream)

    path_to_save = os.path.join(BOT_FOLDER, hashname)
    librosa.output.write_wav(
        path=path_to_save + '.wav',
        y=audio[0],
        sr=sampling_rate)

    sound = AudioSegment.from_wav(path_to_save + '.wav')
    sound.export(path_to_save + '.mp3', format='mp3')

    return path_to_save + '.mp3'


def _get_voices_keyboard(selected: Optional[str] = None):
    keyboard = []
    for voice, _ in VOICES.items():
        if voice == DEFAULT_VOICE:
            continue
        text = voice
        if voice == selected:
            text = f'[ {voice} ]'
        btn = InlineKeyboardButton(text=text, callback_data=voice)
        keyboard.append([btn])
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard)
    return keyboard


async def _get_reply(message: str, user_id: str) -> Tuple[str, str]:
    if user_id in USER_VOICES:
        user_voice = USER_VOICES[user_id]
    else:
        user_voice = START_VOICE

    voice_url = VOICES[user_voice]['url']
    voice_denoiser_strength = VOICES[user_voice]['denoiser_strength']

    inp_dict = {
        "utterance": message,
        'denoiser_strength': voice_denoiser_strength
    }

    payload = json.dumps(inp_dict)
    async with aiohttp.ClientSession(auth=AUTH) as session:
        async with session.post(voice_url, data=payload, headers=HEADERS) as response:
            status = response.status
            if status == http.HTTPStatus.OK:
                responce = await response.read()
            else:
                responce = await response.text()

            return responce, status


@DP.message_handler(commands=['start'])
async def send_kb(message: types.Message):
    """This handler will be called when user sends `/start`"""
    await message.reply(
        f"""Привет. Я озвучу любую отправленную мне фразу на русском языке длиной до {defaults.MAX_UTTERANCE_LENGTH} символов.
         \n Чтобы выбрать голос отправь /voices \n Также, если хочешь сам выставить ударения - добавь знак `+` после требуемой гласной буквы."""
    )


@DP.message_handler(commands=['voices'])
async def send_kb(message: types.Message):
    """This handler will be called if user sends `/voices`"""
    keyboard = _get_voices_keyboard()
    await message.reply(
        "Выбери голос",
        reply_markup=keyboard,
        reply=False
    )


@DP.callback_query_handler(lambda q: q.data in VOICES.keys())
async def send_kb(callback_query: types.CallbackQuery):
    user_id = str(callback_query.from_user.id)
    voice = callback_query.data
    keyboard = _get_voices_keyboard(selected=voice)
    USER_VOICES[user_id] = voice
    await callback_query.message.edit_reply_markup(keyboard)
    await callback_query.message.reply(
        f'<Голос изменен на "{voice}">',
        reply=False
    )


@DP.message_handler()
async def send_reply(message: types.Message):
    """Replies on user message."""
    user_id = str(message.from_user.id)
    responce, status = await _get_reply(message=message.text, user_id=user_id)

    if status == http.HTTPStatus.OK:
        wav_basestring = responce
        path_to_mp3 = get_mp3_path(
            wav_basestring=wav_basestring,
            text=message.text,
            sampling_rate=VOICES[USER_VOICES[user_id]]['sampling_rate'])

        with open(path_to_mp3, 'rb') as audio_file:
            await message.answer_voice(audio_file)
    elif status in (http.HTTPStatus.INTERNAL_SERVER_ERROR, http.HTTPStatus.BAD_REQUEST):
        text = responce
        error_message = f"Error occurred: {text}"
        await message.answer(error_message)


if __name__ == '__main__':
    executor.start_polling(DP, skip_updates=True)
