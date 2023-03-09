import json
from typing import List

from narrativity.common.experiment_config import ExperimentConfig
from narrativity.common.config import Config

class ExperimentConfigList:
    _instance = None

    def __init__(self):
        self._experiment_configs = []

    @staticmethod
    def instantiate() -> None:
        config = Config.instance()
        with open(config.experiments_config_file(), 'rt') as f:
            config = ExperimentConfigList.from_dict(json.load(f))
        ExperimentConfigList._instance = config

    @staticmethod
    def instance() -> "ExperimentConfigList":
        if ExperimentConfigList._instance is None:
            raise Exception('ExperimentConfigList not instantiated, use ExperimentConfigList.instantiate() function first')  # noqa
        return ExperimentConfigList._instance

    def experiment_configs(self) -> List[ExperimentConfig]:
        return self._experiment_configs

    def set_experiment_configs(self, experiment_configs) -> None:
        self._experiment_configs = experiment_configs

    @staticmethod
    def from_dict(val):
        experiments_config_list = ExperimentConfigList()
        experiment_configs = [ExperimentConfig.from_dict(i) for i in val['experiment_configs']]
        experiments_config_list.set_experiment_configs(experiment_configs)
        return experiments_config_list

    def to_dict(self):
        return {
            'experiment_configs': self.experiment_configs(),
        }
