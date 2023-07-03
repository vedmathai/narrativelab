import json
import os
from jadelogs import JadeLogger


class Config:
    _instance = None

    def __init__(self):
        self._pythonpath = None
        self._vitamin_c_data_location = None
        self._vitamin_c_file_mapping = None
        self._nela_quotes_data_location = None
        self._run_configs_file = None
        self._jade_logger = JadeLogger()

    @staticmethod
    def instance():
        if Config._instance is None:
            pythonpath = os.environ['PYTHONPATH']
            config_filepath = os.path.join(pythonpath, 'factuality/common/config.json')
            with open(config_filepath) as f:
                Config._instance = Config.from_dict(json.load(f))
            Config._instance.set_pythonpath(pythonpath)
        return Config._instance
    
    def pythonpath(self):
        return self._pythonpath
    
    def set_pythonpath(self, pythonpath):
        self._pythonpath = pythonpath

    def vitamin_c_data_location(self):
        return self._vitamin_c_data_location

    def set_vitamin_c_data_location(self, vitamin_c_data_location):
        self._vitamin_c_data_location = vitamin_c_data_location

    def vitamin_c_file_mapping(self):
        return self._vitamin_c_file_mapping
    
    def set_vitamin_c_file_mapping(self, vitamin_c_file_mapping):
        self._vitamin_c_file_mapping = vitamin_c_file_mapping

    def nela_quotes_data_location(self):
        filepath = self._jade_logger.file_manager.data_filepath(self._nela_quotes_data_location)
        return filepath
    
    def set_nela_quotes_data_location(self, nela_quotes_data_location):
        self._nela_quotes_data_location = nela_quotes_data_location

    def run_configs_file(self):
        return self._run_configs_file
    
    def run_configs_abs_filepath(self):
        filepath = os.path.join(self.pythonpath(), self.run_configs_file())
        return filepath

    def set_run_configs_file(self, run_configs_file):
        self._run_configs_file = run_configs_file

    @staticmethod
    def from_dict(val):
        config = Config()
        config.set_vitamin_c_data_location(val.get('vitamin_c_data_location'))
        config.set_vitamin_c_file_mapping(val.get('vitamin_c_file_mapping'))
        config.set_nela_quotes_data_location(val.get('nela_quotes_data_location'))
        config.set_run_configs_file(val.get('run_configs_file'))
        return config
