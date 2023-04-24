class RunConfig():
    def __init__(self):
        self._is_train = False
        self._dataset = None

    def is_train(self):
        return self._is_train
    
    def dataset(self):
        return self._dataset
    
    def set_is_train(self, is_train):
        self._is_train = is_train

    def set_dataset(self, dataset):
        self._dataset = dataset

    @staticmethod
    def from_dict(run_config_dict):
        run_config = RunConfig()
        run_config.set_is_train(run_config_dict['is_train'])
        run_config.set_dataset(run_config_dict['dataset'])
        return run_config

    def to_dict(self):
        return {
            'is_train': self.is_train(),
            'dataset': self.dataset(),
        }
