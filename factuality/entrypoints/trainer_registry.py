from factuality.tasks.classification.trainer import ClassificationTrainBase
from factuality.tasks.agreement.trainer import AgreementClassificationTrainBase


class TrainerRegistry:
    _registry = {
        "classification_train_base": ClassificationTrainBase,
        "agreement_classification_train_base": AgreementClassificationTrainBase,
    }

    def get_trainer(self, trainer):
        return self._registry.get(trainer)
