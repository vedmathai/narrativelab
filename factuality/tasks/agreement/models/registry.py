from factuality.tasks.agreement.models.base_model import AgreementClassificationBase
from factuality.tasks.agreement.models.graph_model import AgreementClassificationGraph


class ModelsRegistry:
    _model_dict = {
        "agreement_classification_base": AgreementClassificationBase,
        "agreement_classification_graph": AgreementClassificationGraph,
    }

    def get_model(self, model):
        return ModelsRegistry._model_dict.get(model)
