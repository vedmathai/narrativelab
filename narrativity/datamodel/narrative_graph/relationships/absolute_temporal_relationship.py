from typing import List
import uuid

from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class AbsoluteTemporalRelationship(AbstractRelationship):
    _type = "absolute_temporal_relationship"

    def __init__(self):
        self._preposition = None
        self._narrative_id = None
        self._absolute_temporal_node_id = None
        self._relative_time = []

    def preposition(self):
        return self._preposition

    def relative_time(self):
        return self._relative_time

    def narrative_id(self):
        return self._narrative_id

    def absolute_temporal_node_id(self):
        return self._absolute_temporal_node_id

    def narrative(self):
        return self._narrative_graph.id2narrative_node(self._narrative_id)

    def absolute_temporal_node(self):
        return self._narrative_graph.id2absolute_temporal_node(self._absolute_temporal_node_id)

    def set_preposition(self, preposition):
        self._preposition = preposition

    def set_relative_time(self, relative_time: List[str]) -> None:
        self._relative_time = relative_time

    def set_narrative_id(self, narrative_id: str) -> None:
        self._narrative_id = narrative_id

    def set_absolute_temporal_node_id(self, absolute_temporal_node_id: str) -> None:
        self.absolute_temporal_node_id = absolute_temporal_node_id

    def set_narrative(self, narrative: str) -> None:
        self._narrative_id = narrative.id()

    def set_absolute_temporal_node(self, absolute_temporal_node: str) -> None:
        self._absolute_temporal_node_id = absolute_temporal_node.id()

    def nodes(self) -> List:
        self._nodes = [
            self.narrative(),
            self.absolute_temporal_node(),
        ]
        return super().nodes()

    @staticmethod
    def from_dict(val, narrative_graph):
        temporal_relationship = AbsoluteTemporalRelationship()
        temporal_relationship.set_id(val['id'])
        temporal_relationship.set_narrative_graph(narrative_graph)
        temporal_relationship.set_preposition(val['preposition'])
        temporal_relationship.set_relative_time(val['relative_time'])
        temporal_relationship.set_narrative_id(val['narrative_id'])
        temporal_relationship.set_absolute_temporal_node_id(val['narrative_temporal_node_id'])
        return temporal_relationship
 
    def to_dict(self):
        return {
            "id": self.id(),
            "preposition": self.preposition(),
            "relative_time": self.relative_time(),
            "narrative_id": self.narrative_id(),
            "absolute_temporal_node_id": self.absolute_temporal_node_id(),
        }

    @staticmethod
    def create():
        temporal_relationship = AbsoluteTemporalRelationship()
        temporal_relationship.set_id(str(uuid.uuid4()))
        return temporal_relationship
