from factuality.tasks.quotes.trainer import QuotesClassificationTrainBase


class TrainerRegistry:
    _registry = {
        "quotes_classification_train_base": QuotesClassificationTrainBase,
    }

    def get_trainer(self, trainer):
        print(trainer)
        return self._registry.get(trainer)
