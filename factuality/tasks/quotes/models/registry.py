from factuality.tasks.quotes.models.base_model import QuoteClassificationBase
from factuality.tasks.quotes.models.graph_model import QuoteClassificationGraph


class ModelsRegistry:
    _model_dict = {
        "quote_classification_base": QuoteClassificationBase,
        "quote_classification_graph": QuoteClassificationGraph,
    }

    def get_model(self, model):
        return ModelsRegistry._model_dict.get(model)
