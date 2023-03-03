from typing import List

from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class TemporalRelationship(AbstractRelationship):
    def __init__(self):
        self._preposition = None
        self._temporal_node_ids = []
        self._narrative_id = None
        self._other_narrative_id = None

    def preposition(self):
        return self._preposition

    def temporal_node_ids(self):
        return self._temporal_node_ids

    def temporal_nodes(self):
        return [self._narrative_graph.id2temporal_node(i) for i in self._temporal_node_ids]

    def narrative_id(self):
        return self._narrative_id

    def other_narrative_id(self):
        return self._other_narrative_id

    def narrative(self):
        return self._narrative_graph.id2narrative_node(self._narrative)

    def other_narrative(self):
        return self._narrative_graph.id2narrative_node(self._narrative)

    def set_preposition(self, preposition):
        self._preposition = preposition

    def set_temporal_node_ids(self, temporal_node_ids: List[str]) -> None:
        self._temporal_node_ids = temporal_node_ids

    def set_narrative_id(self, narrative_id: str) -> None:
        self._narrative_id = narrative_id

    def set_other_narrative_id(self, narrative_id: str) -> None:
        self._other_narrative_id = narrative_id

    @staticmethod
    def from_dict(val, narrative_graph):
        temporal_relationship = TemporalRelationship()
        temporal_relationship.set_id(val['id'])
        temporal_relationship.set_narrative_graph(narrative_graph)
        temporal_relationship.set_preposition(val['preposition'])
        temporal_relationship.set_temporal_node_ids(val['temporal_node_ids'])
        temporal_relationship.set_narrative_id(val['narrative_id'])
        temporal_relationship.set_other_narrative_id(val['other_narrative_id'])
        return temporal_relationship
 
    def to_dict(self):
        return {
            "id": self.id(),
            "preposition": self.preposition(),
            "time_nodes": self.time_nodes(),
            "narrative": self.narrative(),
            "other_narrative": self.other_relative(),
        }
