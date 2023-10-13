from factuality.tasks.classification.models.base_model import ClassificationBase
from factuality.tasks.classification.models.graph_model import ClassificationGraph


class ModelsRegistry:
    _model_dict = {
        "classification_base": ClassificationBase,
        "classification_graph": ClassificationGraph,
    }

    def get_model(self, model):
        print(model)
        return ModelsRegistry._model_dict.get(model)
