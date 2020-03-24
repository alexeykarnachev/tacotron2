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
START_VOICE = VOICES[config['default_voice']]
USER_VOICES = defaultdict(lambda: START_VOICE)


def get_mp3_path(wav_basestring, text, sampling_rate):
    hashname = hash(text)
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
    user_voice = USER_VOICES[user_id]
    voice_url = VOICES[user_voice]['url']

    inp_dict = {
        "utterance": message
    }

    payload = json.dumps(inp_dict)
    async with aiohttp.ClientSession(auth=AUTH) as session:
        async with session.post(voice_url, data=payload, headers=HEADERS) as response:
            status = response.status
            bytecode = response.content.read()
            return bytecode, status


@DP.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    """This handler will be called when user sends `/start` or command"""
    keyboard = _get_voices_keyboard()
    await message.reply(
        "Выбери голос",
        reply_markup=keyboard,
        reply=False
    )


@DP.callback_query_handler(lambda q: q.data in VOICES.keys())
async def send_welcome(callback_query: types.CallbackQuery):
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
    text, status = await _get_reply(message=message.text, user_id=user_id)

    if status == 200:
        wav_basestring = text
        path_to_mp3 = get_mp3_path(
            wav_basestring=wav_basestring,
            text=message.text,
            sampling_rate=VOICES[USER_VOICES[user_id]]['sampling_rate'])

        with open(path_to_mp3, 'r') as audio_file:
            await message.answer_audio(audio=audio_file)
    elif status == 400:
        reply = json.loads(text)
        error_message = f"Bad request: {reply['utterance']}"
        await message.answer(error_message)


if __name__ == '__main__':
    executor.start_polling(DP, skip_updates=True)
