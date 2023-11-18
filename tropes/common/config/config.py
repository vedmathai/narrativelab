import json
import os
from jadelogs import JadeLogger


class Config:
    _instance = None

    def __init__(self):
        self._pythonpath = None
        self._tv_tropes_data_location = None
        self._experiment_type = None
        self._run_configs_file = None
        self._wikipedia_data_location = None
        self._jade_logger = JadeLogger()

    @staticmethod
    def instance():
        if Config._instance is None:
            jade_logger = JadeLogger()
            config_filepath = jade_logger.file_manager.code_filepath('narrativelab/tropes/common/config/config.json')
            with open(config_filepath) as f:
                Config._instance = Config.from_dict(json.load(f))
        return Config._instance
    
    def pythonpath(self):
        return self._pythonpath

    def tv_tropes_data_location(self):
        location = self._jade_logger.file_manager.data_filepath(self._tv_tropes_data_location)
        return location

    def set_tv_tropes_data_location(self, tv_tropes_data_location):
        self._tv_tropes_data_location = tv_tropes_data_location

    def wikipedia_data_location(self):
        location = self._jade_logger.file_manager.data_filepath(self._wikipedia_data_location)
        return location

    def set_wikipedia_data_location(self, wikipedia_data_location):
        self._wikipedia_data_location = wikipedia_data_location

    def run_configs_file(self):
        return self._run_configs_file
    
    def run_configs_abs_filepath(self):
        filepath = os.path.join(self.pythonpath(), self.run_configs_file())
        return filepath

    def experiment_type(self):
        return self._experiment_type

    def model_save_location(self):
        location = self._jade_logger.file_manager.data_filepath(self._model_save_location)
        return location
    
    def set_pythonpath(self, pythonpath):
        self._pythonpath = pythonpath

    def set_run_configs_file(self, run_configs_file):
        self._run_configs_file = run_configs_file

    def set_experiment_type(self, experiment_type):
        self._experiment_type = experiment_type

    def set_model_save_location(self, model_save_location):
        self._model_save_location = model_save_location

    @staticmethod
    def from_dict(val):
        config = Config()
        config.set_tv_tropes_data_location(val.get('tv_tropes_data_location'))
        config.set_experiment_type(val.get('experiment_type'))
        config.set_run_configs_file(val.get('run_configs_file'))
        config.set_model_save_location(val.get('model_save_location'))
        config.set_wikipedia_data_location(val.get('wikipedia_data_location'))
        return config
