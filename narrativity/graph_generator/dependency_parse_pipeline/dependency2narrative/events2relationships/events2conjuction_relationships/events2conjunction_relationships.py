from narrativity.datamodel.narrative_graph.relationships.contradictory_relationship import ContradictoryRelationship
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.events2relationships.abstract_events2relationships import AbstractEvents2Relationships


class Events2ConjunctionRelationships(AbstractEvents2Relationships):  
    def __init__(self):
        super().__init__()

    def load(self):
        super().load()
 
    def extract(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        extraction_fns = [
            self._extract_but_like_contradiction,
            self._extract_however_like_contradiction,
            self._extract_though_like_contradiction,
            self._extract_and_like_conjunction,
        ]
        for fn in extraction_fns:
            fn(narrative_1, narrative_2, phrase_connector, narrative_graph)
    
    def _extract_but_like_contradiction(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        if phrase_connector.connector_type() in ['but_like_contradiction']:
            contradictory_relationship = ContradictoryRelationship.create()
            contradictory_relationship.set_narrative_graph(narrative_graph)
            contradictory_relationship.set_narrative_1(narrative_2)
            contradictory_relationship.set_narrative_2(narrative_1)
            narrative_1.add_contradictory_in_relationship(contradictory_relationship)
            narrative_2.add_contradictory_out_relationship(contradictory_relationship)
            contradictory_relationship.set_mark(phrase_connector.connector_text().text())
            narrative_graph.add_contradictory_relationship(contradictory_relationship)

    def _extract_however_like_contradiction(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        if phrase_connector.connector_type() in ['however_like_contradiction']:
            contradictory_relationship = ContradictoryRelationship.create()
            contradictory_relationship.set_narrative_graph(narrative_graph)
            contradictory_relationship.set_narrative_1(narrative_2)
            contradictory_relationship.set_narrative_2(narrative_1)
            narrative_1.add_contradictory_in_relationship(contradictory_relationship)
            narrative_2.add_contradictory_out_relationship(contradictory_relationship)
            contradictory_relationship.set_mark(phrase_connector.connector_text().text())
            narrative_graph.add_contradictory_relationship(contradictory_relationship)

    def _extract_though_like_contradiction(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        if phrase_connector.connector_type() in ['though_like_contradiction']:
            contradictory_relationship = ContradictoryRelationship.create()
            contradictory_relationship.set_narrative_graph(narrative_graph)
            contradictory_relationship.set_narrative_1(narrative_2)
            contradictory_relationship.set_narrative_2(narrative_1)
            narrative_1.add_contradictory_in_relationship(contradictory_relationship)
            narrative_2.add_contradictory_out_relationship(contradictory_relationship)
            contradictory_relationship.set_mark(phrase_connector.connector_text().text())
            narrative_graph.add_contradictory_relationship(contradictory_relationship)

    def _extract_and_like_conjunction(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        if phrase_connector.connector_type() in ['and_like_relationship']:
            and_relationship = ContradictoryRelationship.create()
            and_relationship.set_narrative_graph(narrative_graph)
            and_relationship.set_narrative_1(narrative_2)
            and_relationship.set_narrative_2(narrative_1)
            narrative_1.add_and_like_relationship(and_relationship)
            narrative_2.add_and_like_relationship(and_relationship)
            if phrase_connector.connector_text() is not None:
                and_relationship.set_mark(phrase_connector.connector_text().text())
            narrative_graph.add_and_like_relationship(and_relationship)
