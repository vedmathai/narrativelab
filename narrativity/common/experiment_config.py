
class ExperimentConfig:
    _instance = None

    def __init__(self):
        pass

    @staticmethod
    def from_dict(val):
        experiments_config = ExperimentConfig()
        return experiments_config

    def to_dict(self):
        return {
        }
