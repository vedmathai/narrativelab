import json
from jadelogs import JadeLogger


class Config:
    _instance = None

    def __init__(self):
        self._ecb_path = None
        self._jade_logger = JadeLogger()

    @staticmethod
    def instance():
        if Config._instance is None:
            jade_logger = JadeLogger()
            config_filepath = jade_logger.file_manager.code_filepath('narrativelab/coreference/common/config/config.json')
            with open(config_filepath) as f:
                Config._instance = Config.from_dict(json.load(f))
        return Config._instance
    
    def pythonpath(self):
        return self._pythonpath

    def ecb_path(self):
        location = self._jade_logger.file_manager.data_filepath(self._ecb_path)
        print(location)
        return location

    def set_ecb_path(self, ecb_path):
        self._ecb_path = ecb_path

    @staticmethod
    def from_dict(val):
        config = Config()
        config.set_ecb_path(val.get('ecb_path'))
        return config
