import json

from coreference.common.config.config import Config
from coreference.common.config.run_configs.run_configs_datamodel.run_configs import RunConfigs


class RunConfigsLoader:
    def __init__(self):
        self._config = Config.instance()
        self._run_configs = None

    def load(self):
        with open(self._config.run_configs_abs_filepath(), 'rt') as f:
            run_configs_dict = json.load(f)
            self._run_configs = RunConfigs.from_dict(run_configs_dict)
        return run_configs_dict

    def run_configs(self):
        return self._run_configs

