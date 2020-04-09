from pathlib import Path
from setuptools import find_packages, setup

THIS_DIR = Path(__file__).parent


def get_version(filename):
    from re import findall
    with open(filename) as f:
        metadata = dict(findall("__([a-z]+)__ = '([^']+)'", f.read()))
    return metadata['version']


setup(
    name='tacotron2',
    version=get_version('tacotron2/__init__.py'),
    description='Tacotron2 TTS Model',
    packages=find_packages(exclude=['test', 'test.*']),
    install_requires=[
        "torch==1.3.1",
        "scikit-learn==0.21.3",
        "pandas==0.25.1",
        "future==0.17.1",
        "tqdm==4.36.1",
        "librosa==0.7.2",
        "pydub==0.23.1",
        "pillow==7.0.0",
        "matplotlib==3.1.3",
        "num2words==0.5.10",
        "rnnmorph==0.4.0",
        "lxml==4.5.0",
        "transformers==2.4.1",
        "unidecode==1.1.1",
        "pytest==5.3.5",
        "flask==1.1.1",
        "gunicorn==19.9.0",
        "marshmallow==3.5.0",
        "flasgger==0.9.4",
        "apispec==3.3.0",
        "apispec-webframeworks==0.5.2",
        "flask-basicauth==0.2.0",
        "IPython==7.13.0",
        "aiogram==2.6.1",
        "aiohttp==3.6.2",
        "inflect==4.1.0",
        "razdel==0.5.0",
        "pymorphy2==0.8",
        "requests==2.23.0",
        "beautifulsoup4==4.8.2",
        "gdown==3.10.2",
        "tokenizers==0.5.0",
        "dawg @ http://github.com/pytries/DAWG/tarball/master",
        "russian_g2p @ http://github.com/nsu-ai/russian_g2p/tarball/master",
        "rnd_utilities @ git+https://bitbucket.org/just-ai/rnd_utilities/get/master.tar.gz",
        "rnd_datasets @ git+https://bitbucket.org/just-ai/rnd_datasets/get/master.tar.gz",
        "pytorch-lightning @ https://github.com/alexeykarnachev/pytorch-lightning/archive/master.zip"
    ],
    package_dir={'tacotron2': 'tacotron2'},
    package_data={'tacotron2': ['tokenizers/data/*']}
)
