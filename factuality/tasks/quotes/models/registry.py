from factuality.tasks.quotes.models.base_model import QuoteClassificationBase


class ModelsRegistry:
    _model_dict = {
        "quote_classification_base": QuoteClassificationBase
    }

    def get_model(self, model):
        return ModelsRegistry._model_dict.get(model)
