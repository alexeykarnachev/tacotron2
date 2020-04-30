from tacotron2.app.syntesis import utilities

import os
os.environ["APP_CONFIG"] = './app_config.json'
APP = utilities.prepare()

if __name__ == '__main__':
    APP.run()

