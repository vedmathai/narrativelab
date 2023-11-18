import json
import os
from jadelogs import JadeLogger

class Config:
    _instance = None

    @staticmethod
    def instance():
        if Config._instance is None:
            jade_logger = JadeLogger()
            config_filepath = jade_logger.file_manager.code_filepath('narrativelab/summarizer/common/config/config.json')
            with open(config_filepath) as f:
                Config._instance = Config.from_dict(json.load(f))
        return Config._instance

    def __init__(self):
        self._memsum_arxiv_model_path = None
        self._memsum_vocab_path = None
        self._jade_logger = JadeLogger()

    def memsum_vocab_path(self):
        location = self._jade_logger.file_manager.data_filepath(self._memsum_vocab_path)
        location = self._memsum_vocab_path
        return location
    
    def memsum_arxiv_model_path(self):
        location = self._jade_logger.file_manager.data_filepath(self._memsum_arxiv_model_path)
        location = self._memsum_arxiv_model_path
        return location
    
    def set_memsum_arxiv_model_path(self, memsum_arxiv_model_path):
        self._memsum_arxiv_model_path = memsum_arxiv_model_path

    def set_memsum_vocab_path(self, memsum_vocab_path):
        self._memsum_vocab_path = memsum_vocab_path

    @staticmethod
    def from_dict(val):
        config = Config()
        config.set_memsum_arxiv_model_path(val.get('memsum_arxiv_model_path'))
        config.set_memsum_vocab_path(val.get('memsum_vocab_path'))
        return config
