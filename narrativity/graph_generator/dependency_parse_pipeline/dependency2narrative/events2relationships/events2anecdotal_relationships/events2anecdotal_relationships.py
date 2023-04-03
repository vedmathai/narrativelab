from narrativity.datamodel.narrative_graph.relationships.anecdotal_relationship import AnecdotalRelationship


class Events2AnecdotalRelationships:

    def load(self):
        pass

    def extract(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        is_relationship = self._extract_anecdote2relationship(narrative_1, narrative_2, phrase_connector, narrative_graph)
        if is_relationship is False:
            self._add_anecdote2narrative(narrative_1, narrative_2, phrase_connector, narrative_graph)

    def _extract_anecdote2relationship(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        is_relationship = False
        relationship_fns = [
            narrative_2.temporal_event_in_relationships,
            narrative_2.temporal_event_out_relationships,
            narrative_2.causal_out_relationships,
            narrative_2.causal_in_relationships,
            narrative_2.contradictory_out_relationships,
            narrative_2.contradictory_in_relationships,
        ]
        relationships = []
        for fn in relationship_fns:
            relationships.extend(fn())
        for relationship in relationships:
            is_relationship = True
            self._add_anecdote2relationship(narrative_1, relationship, phrase_connector, narrative_graph)
        return is_relationship

    def _add_anecdote2relationship(self, narrative, relationship, phrase_connector, narrative_graph):
        anecdotal_relationship = AnecdotalRelationship.create()
        anecdotal_relationship.set_is_to_relationship(True)
        anecdotal_relationship.set_narrative_graph(narrative_graph)
        anecdotal_relationship.set_narrative_1(narrative)
        anecdotal_relationship.set_relationship(relationship)
        relationship.add_anecdotal_in_relationship(anecdotal_relationship)
        narrative.add_anecdotal_out_relationship(anecdotal_relationship)
        narrative_graph.add_anecdotal_relationship(anecdotal_relationship)

    def _add_anecdote2narrative(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        anecdotal_relationship = AnecdotalRelationship.create()
        anecdotal_relationship.set_narrative_graph(narrative_graph)
        anecdotal_relationship.set_is_to_relationship(False)
        anecdotal_relationship.set_narrative_1(narrative_1)
        anecdotal_relationship.set_narrative_2(narrative_2)
        narrative_1.add_anecdotal_out_relationship(anecdotal_relationship)
        narrative_2.add_anecdotal_in_relationship(anecdotal_relationship)
        narrative_graph.add_anecdotal_relationship(anecdotal_relationship)
