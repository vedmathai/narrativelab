from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.events2relationships.events2causal_relationships.events2causal_relationships import Events2CausalRelationships
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.events2relationships.events2temporal_relationships.events2temporal_event_relationships import Events2TemporalEventRelationships
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.events2relationships.events2conjuction_relationships.events2conjunction_relationships import Events2ConjunctionRelationships
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.events2relationships.events2spoken_relationships.events2spoken_relationships import Events2SpokenRelationships


class Events2Relationships:
    def __init__(self):
        self._events2causal_relationships = Events2CausalRelationships()
        self._events2temporal_event_relationships = Events2TemporalEventRelationships()
        self._events2conjunction_relationships = Events2ConjunctionRelationships()
        self._events2spoken_relationships = Events2SpokenRelationships()

    def load(self):
        self._events2causal_relationships.load()
        self._events2temporal_event_relationships.load()
        self._events2conjunction_relationships.load()
        self._events2spoken_relationships.load()

    def extract(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        extractor_classes = [
            self._events2causal_relationships,
            self._events2conjunction_relationships,
            self._events2spoken_relationships,
            self._events2temporal_event_relationships,
        ]
        for extractor_class in extractor_classes:
            extractor_class.extract(narrative_1, narrative_2, phrase_connector, narrative_graph)