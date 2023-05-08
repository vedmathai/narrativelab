from narrativity.datamodel.narrative_graph.nodes.narrative_node import NarrativeNode
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.verb2event.verb2actions.verb2actions import Verb2Actions
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.verb2event.verb2actors.verb2actors import Verb2Actors
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.verbaux2objects.verbaux2indirect_objects import VerbAux2IndirectObjects
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.verb2event.verb2objects.verb2direct_objects import Verb2DirectObjects
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.verbaux2locations.verbaux2locations import VerbAux2Locations
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.verbaux2absolute_time.verbaux2absolute_time import VerbAux2AbsoluteTimes
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import get_all_children_tokens


class Verb2Event:
    def __init__(self):
        self._verb2actions = Verb2Actions()
        self._verb2indirect_objects = VerbAux2IndirectObjects()
        self._verb2direct_objects = Verb2DirectObjects()
        self._verb2actors = Verb2Actors()
        self._verb2locations = VerbAux2Locations()
        self._verb2absolute_times = VerbAux2AbsoluteTimes()
        self._converters = [
            self._verb2actions,
            self._verb2indirect_objects,
            self._verb2direct_objects,
            self._verb2actors,
            self._verb2locations,
            self._verb2absolute_times
        ]

    def load(self):
        for converter in self._converters:
            converter.load()

    def convert(self, verb, narrative_graph):
        narrative_node = NarrativeNode.create()
        all_children_tokens = get_all_children_tokens(verb)
        for converter in self._converters:
            converter.convert(verb, all_children_tokens, narrative_node, narrative_graph)
        narrative_graph.add_narrative_node(narrative_node)
        narrative_node.set_narrative_graph(narrative_graph)
        return narrative_node
