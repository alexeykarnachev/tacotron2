from pathlib import Path
import gdown
import argparse

_GOOGLE_DRIVE_PATH = "https://drive.google.com/uc?id={}"

parser = argparse.ArgumentParser()
parser.add_argument('--file_id', type=str, help='google drive file id') # 1AjWop6_-ZErmbklE83U5pXpjTYgSM3vr
args = parser.parse_args()


if __name__ == '__main__':
    accents_path = Path(__file__).parent.parent / 'tokenizers/data/accents.json'

    if not accents_path.is_file():
        gdown.download(_GOOGLE_DRIVE_PATH.format(args.file_id), str(accents_path.absolute()))