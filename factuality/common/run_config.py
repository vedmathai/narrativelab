class RunConfig:
    def __init__(self):
        self._id = None
        self._is_train = None
        self._trainer = None
        self._llm = None,
        self._epochs = None
        self._dataset = None
        self._model_type = None
        self._featurizer = None

    def id(self):
        return self._id

    def is_train(self):
        return self._is_train
    
    def trainer(self):
        return self._trainer
    
    def llm(self):
        return self._llm
    
    def epochs(self):
        return self._epochs
    
    def dataset(self):
        return self._dataset
    
    def model_type(self):
        return self._model_type
    
    def featurizer(self):
        return self._featurizer
    
    def set_id(self, id):
        self._id = id
    
    def set_is_train(self, is_train):
        self._is_train = is_train

    def set_trainer(self, trainer):
        self._trainer = trainer

    def set_llm(self, llm):
        self._llm = llm

    def set_epochs(self, epochs):
        self._epochs = epochs

    def set_dataset(self, dataset):
        self._dataset = dataset

    def set_model_type(self, model_type):
        self._model_type = model_type

    def set_featurizer(self, featurizer):
        self._featurizer = featurizer

    @staticmethod
    def from_dict(val):
        run_config = RunConfig()
        run_config.set_id(val['id'])
        run_config.set_is_train(val['is_train'])
        run_config.set_trainer(val['trainer'])
        run_config.set_llm(val['llm'])
        run_config.set_epochs(val['epochs'])
        run_config.set_dataset(val['dataset'])
        run_config.set_model_type(val['model_type'])
        run_config.set_featurizer(val['featurizer'])
        return run_config
    
    def to_dict(self):
        return {
            "id": self.id(),
            "is_train": self.is_train(),
            "trainer": self.trainer(),
            "llm": self.llm(),
            "epochs": self.epochs(),
            "dataset": self.dataset(),
            "model_type": self.model_type(),
            "featurizer": self.featurizer(),
        }



