from narrativity.datamodel.narrative_graph.relationships.temporal_event_relationship import TemporalEventRelationship
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.events2relationships.abstract_events2relationships import AbstractEvents2Relationships


class Events2TemporalEventRelationships(AbstractEvents2Relationships):  
    def __init__(self):
        super().__init__()

    def load(self):
        super().load()
 
    def extract(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        extraction_fns = [
            self._after_like_temporal_relationship,
        ]
        for fn in extraction_fns:
            fn(narrative_1, narrative_2, phrase_connector, narrative_graph)
    
    def _after_like_temporal_relationship(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        if phrase_connector.connector_type() in ['after_like_temporal_relationship']:
            temporal_event_relationship = TemporalEventRelationship.create()
            temporal_event_relationship.set_narrative_graph(narrative_graph)
            temporal_event_relationship.set_narrative_1(narrative_1)
            temporal_event_relationship.set_narrative_2(narrative_2)
            narrative_1.add_temporal_event_out_relationship(temporal_event_relationship)
            narrative_2.add_temporal_event_in_relationship(temporal_event_relationship)
            temporal_event_relationship.set_preposition(phrase_connector.connector_text().text())
            narrative_graph.add_temporal_event_relationship(temporal_event_relationship)
