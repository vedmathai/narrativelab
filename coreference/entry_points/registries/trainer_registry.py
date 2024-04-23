from coreference.trainer.trainer import CoreferenceClassificationTrain


class TrainerRegistry:
    _registry = {
        'coreference_classification': CoreferenceClassificationTrain,
    }

    def get_trainer(self, trainer):
        return self._registry.get(trainer)
