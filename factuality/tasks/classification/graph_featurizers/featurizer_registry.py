from factuality.tasks.classification.graph_featurizers.narrative_graph_featurizer_single import NarrativeGraphFeaturizer
from factuality.tasks.classification.graph_featurizers.dependency_parse_featurizer import DependencyParseFeaturizer
from factuality.tasks.classification.graph_featurizers.semantic_role_labelling_featurizer import SRLParseFeaturizer
from factuality.tasks.classification.graph_featurizers.amr_featurizer import AMRParseFeaturizer


class FeaturizerRegistry:
    _registry = {
        'dependency': DependencyParseFeaturizer,
        'narrative': NarrativeGraphFeaturizer,
        'srl': SRLParseFeaturizer,
        'amr': AMRParseFeaturizer,
    }

    def get_featurizer(self, featurizer_name):
        return self._registry[featurizer_name]()
