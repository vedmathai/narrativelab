from narrativity.datamodel.narrative_graph.nodes.narrative_node import NarrativeNode
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.aux2state.aux2subjects.aux2subjects import Aux2Subjects
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.aux2state.aux2description.aux2description import Aux2Description
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import get_all_children_tokens


class Aux2State:
    def __init__(self):
        self._aux2subjects = Aux2Subjects()
        self._aux2description = Aux2Description()

    def load(self):
        self._aux2subjects.load()
        self._aux2description.load()

    def convert(self, aux, narrative_graph):
        narrative_node = NarrativeNode.create()
        all_children_tokens = get_all_children_tokens(aux)
        self._aux2subjects.convert(aux, all_children_tokens, narrative_node, narrative_graph)
        self._aux2description.convert(aux, all_children_tokens, narrative_node, narrative_graph)
        auxiliary_whole_text = self._resolve_auxiliary(aux)
        narrative_graph.add_narrative_node(narrative_node)
        narrative_node.set_narrative_graph(narrative_graph)

    def _resolve_auxiliary(self, aux):
        whole_text = aux.text()
        return whole_text

