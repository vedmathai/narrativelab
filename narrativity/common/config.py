import json


config_files = {
    'local': 'narrativity/common/configs/local_config.json',
}

class Config:
    _instance = None

    def __init__(self):
        self._keyring_file = None
        self._experiments_config_file = None

    @staticmethod
    def instantiate(tier) -> None:
        config_file = config_files.get(tier)
        with open(config_file, 'rt') as f:
            config = Config.from_dict(json.load(f))
        Config._instance = config

    @staticmethod
    def instance() -> "Config":
        if Config._instance is None:
            raise Exception('Config not instantiated, use Config.instantiate() function first')  # noqa
        return Config._instance

    def experiments_config_file(self) -> str:
        return self._experiments_config_file

    def set_experiments_config_file(self, experiments_config_file) -> str:
        self._experiments_config_file = experiments_config_file

    def keyring_file(self) -> str:
        return self._keyring_file

    def set_keyring_file(self, keyring_file) -> str:
        self._keyring_file = keyring_file

    @staticmethod
    def from_dict(val):
        config = Config()
        config.set_keyring_file(val.get('keyring_file'))
        config.set_experiments_config_file(val.get('experiments_config_file'))
        return config

    def to_dict(self):
        return {
            'keyring_file': self.keyring_file(),
            'experiments_config_file': self.experiments_config_file(),
        }
