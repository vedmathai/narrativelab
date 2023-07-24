from factuality.tasks.quotes.trainer import QuotesClassificationTrainBase
from factuality.tasks.agreement.trainer import AgreementClassificationTrainBase


class TrainerRegistry:
    _registry = {
        "quotes_classification_train_base": QuotesClassificationTrainBase,
        "agreement_classification_train_base": AgreementClassificationTrainBase,
    }

    def get_trainer(self, trainer):
        return self._registry.get(trainer)
