from narrativity.datamodel.narrative_graph.nodes.narrative_node import NarrativeNode
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.verb2event.verb2actions.verb2actions import Verb2Actions
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.verb2event.verb2actors.verb2actors import Verb2Actors
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.verb2event.verb2objects.verb2indirect_objects import Verb2IndirectObjects
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.verb2event.verb2objects.verb2direct_objects import Verb2DirectObjects
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import get_all_children_tokens


class Verb2Event:
    def __init__(self):
        self._verb2actions = Verb2Actions()
        self._verb2indirect_objects = Verb2IndirectObjects()
        self._verb2direct_objects = Verb2DirectObjects()
        self._verb2actors = Verb2Actors()

    def load(self):
        self._verb2actions.load()
        self._verb2indirect_objects.load()
        self._verb2direct_objects.load()
        self._verb2actors.load()

    def convert(self, verb, narrative_graph):
        narrative_node = NarrativeNode.create()
        all_children_tokens = get_all_children_tokens(verb)
        self._verb2actors.convert(verb, all_children_tokens, narrative_node, narrative_graph)
        self._verb2actions.convert(verb, all_children_tokens, narrative_node, narrative_graph)
        self._verb2indirect_objects.convert(verb, all_children_tokens, narrative_node, narrative_graph)
        self._verb2direct_objects.convert(verb, all_children_tokens, narrative_node, narrative_graph)
        narrative_graph.add_narrative_node(narrative_node)
        narrative_node.set_narrative_graph(narrative_graph)
