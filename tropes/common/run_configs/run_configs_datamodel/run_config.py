class RunConfig():
    def __init__(self):
        self._id = None
        self._is_train = False
        self._llm = None
        self._epochs = None
        self._dataset = None
        self._trainer = None

    def id(self):
        return self._id

    def is_train(self):
        return self._is_train
    
    def llm(self):
        return self._llm
    
    def epochs(self):
        return self._epochs
    
    def dataset(self):
        return self._dataset
    
    def trainer(self):
        return self._trainer
    
    def set_id(self, id):
        self._id = id

    def set_is_train(self, is_train):
        self._is_train = is_train

    def set_llm(self, llm):
        self._llm = llm

    def set_epochs(self, epochs):
        self._epochs = epochs

    def set_dataset(self, dataset):
        self._dataset = dataset

    def set_trainer(self, trainer):
        self._trainer = trainer

    @staticmethod
    def from_dict(run_config_dict):
        run_config = RunConfig()
        run_config.set_id(run_config_dict['id'])
        run_config.set_is_train(run_config_dict['is_train'])
        run_config.set_llm(run_config_dict['llm'])
        run_config.set_epochs(run_config_dict['epochs'])
        run_config.set_dataset(run_config_dict['dataset'])
        run_config.set_trainer(run_config_dict['trainer'])
        return run_config

    def to_dict(self):
        return {
            'id': self.id(),
            'is_train': self.is_train(),
            'llm': self.llm(),
            'epochs': self.epochs(),
            'dataset': self.dataset(),
            'forward_type': self.forward_type(),
            'trainer': self.trainer(),
        }
