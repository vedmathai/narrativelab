from narrativity.datamodel.narrative_graph.relationships.prep_relationship import PrepRelationship
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.events2relationships.abstract_events2relationships import AbstractEvents2Relationships


class Events2PrepRelationships(AbstractEvents2Relationships):  
    def __init__(self):
        super().__init__()

    def load(self):
        super().load()
 
    def extract(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        extraction_fns = [
            self._prep_relationship,
        ]
        for fn in extraction_fns:
            fn(narrative_1, narrative_2, phrase_connector, narrative_graph)
    
    def _prep_relationship(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        if phrase_connector.connector_type() in ['prep_relationship']:
            prep_relationship = PrepRelationship.create()
            prep_relationship.set_narrative_graph(narrative_graph)
            prep_relationship.set_narrative_1(narrative_1)
            prep_relationship.set_narrative_2(narrative_2)
            narrative_1.add_prep_out_relationship(prep_relationship)
            narrative_2.add_prep_in_relationship(prep_relationship)
            prep_relationship.set_preposition(phrase_connector.connector_text().text())
            narrative_graph.add_prep_relationship(prep_relationship)
