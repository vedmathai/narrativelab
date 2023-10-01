from tropes.trainer.trope_classification.trainers.train import TropeClassificationTrain


class TrainerRegistry:
    _registry = {
        'trope_classification': TropeClassificationTrain,
    }

    def get_trainer(self, trainer):
        return self._registry.get(trainer)
