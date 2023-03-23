from narrativity.datamodel.narrative_graph.relationships.causal_relationship import CausalRelationship
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.events2relationships.abstract_events2relationships import AbstractEvents2Relationships


class Events2CausalRelationships(AbstractEvents2Relationships):  
    def __init__(self):
        super().__init__()

    def load(self):
        super().load()
        self._forwards_causal_marks = self._extraction_paths.common().key2word_list("$$forwards_causal_marks$$")
        self._backwards_causal_marks = self._extraction_paths.common().key2word_list("$$backwards_causal_marks$$")
 
    def extract(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        if phrase_connector.connector_type() == 'causation':
            causal_relationship = CausalRelationship.create()
            causal_relationship.set_narrative_graph(narrative_graph)
            if phrase_connector.connector_text().text() in self._backwards_causal_marks:
                causal_relationship.set_narrative_1(narrative_2)
                causal_relationship.set_narrative_2(narrative_1)
                narrative_1.add_causal_in_relationship(causal_relationship)
                narrative_2.add_causal_out_relationship(causal_relationship)
            if phrase_connector.connector_text().text() in self._forwards_causal_marks:
                causal_relationship.set_narrative_1(narrative_1)
                causal_relationship.set_narrative_2(narrative_2)
                narrative_1.add_causal_out_relationship(causal_relationship)
                narrative_2.add_causal_in_relationship(causal_relationship)
            causal_relationship.set_mark(phrase_connector.connector_text().text())
            narrative_graph.add_causal_relationship(causal_relationship)