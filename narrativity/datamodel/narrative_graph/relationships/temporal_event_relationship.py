from typing import List
import uuid

from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class TemporalEventRelationship(AbstractRelationship):
    _type = "temporal_event_relationship"

    def __init__(self):
        self._preposition = None
        self._narrative_1_id = None
        self._narrative_2_id = None
        self._relative_time = []
        self._anecdotal_in_relationship_ids = []

    def display_name(self):
        narrative_1 = self.narrative_1().display_name()
        narrative_2 = self.narrative_2().display_name()
        preposition = self.preposition()
        temporal_relationship = "{} happens {} {}".format(narrative_1, preposition, narrative_2)
        return temporal_relationship

    def preposition(self):
        return self._preposition

    def relative_time(self):
        return self._relative_time

    def narrative_1_id(self):
        return self._narrative_1_id

    def narrative_2_id(self):
        return self._narrative_2_id

    def narrative_1(self):
        return self._narrative_graph.id2narrative_node(self._narrative_1_id)

    def narrative_2(self):
        return self._narrative_graph.id2narrative_node(self._narrative_2_id)

    def anecdotal_in_relationship_ids(self):
        return self._anecdotal_in_relationship_ids
    
    def anecdotal_in_relationships(self):
        return [self._narrative_graph.id2anecdotal_relationship(i) for i in self._anecdotal_in_relationship_ids]

    def set_preposition(self, preposition):
        self._preposition = preposition

    def set_relative_time(self, relative_time: List[str]) -> None:
        self._relative_time = relative_time

    def set_narrative_1_id(self, narrative_1_id: str) -> None:
        self._narrative_1_id = narrative_1_id

    def set_narrative_2_id(self, narrative_2_id: str) -> None:
        self._narrative_2_id = narrative_2_id

    def set_narrative_1(self, narrative_1: str) -> None:
        self._narrative_1_id = narrative_1.id()

    def set_narrative_2(self, narrative_2: str) -> None:
        self._narrative_2_id = narrative_2.id()

    def add_anecdotal_in_relationship_id(self, anecdotal_in_relationship_id):
        self._anecdotal_in_relationship_ids.append(anecdotal_in_relationship_id)

    def add_anecdotal_in_relationship(self, anecdotal_in_relationship):
        self._anecdotal_in_relationship_ids.append(anecdotal_in_relationship.id())

    def set_anecdotal_in_relationship_ids(self, anecdotal_in_relationship_ids):
        self._anecdotal_in_relationship_ids = anecdotal_in_relationship_ids

    def nodes(self):
        self._nodes = [
            self.narrative_1(),
            self.narrative_2(),
            *self.anecdotal_in_relationships(),
        ]
        return super().nodes()
    @staticmethod
    def from_dict(val, narrative_graph):
        temporal_relationship = TemporalEventRelationship()
        temporal_relationship.set_id(val['id'])
        temporal_relationship.set_narrative_graph(narrative_graph)
        temporal_relationship.set_preposition(val['preposition'])
        temporal_relationship.set_relative_time(val['relative_time'])
        temporal_relationship.set_narrative_1_id(val['narrative_1_id'])
        temporal_relationship.set_narrative_2_id(val['narrative_2_id'])
        temporal_relationship.set_anecdotal_in_relationship_ids(val['anecdotal_in_relationship_ids'])
        return temporal_relationship
 
    def to_dict(self):
        return {
            "id": self.id(),
            "display_name": self.display_name(),
            "preposition": self.preposition(),
            "relative_time": self.relative_time(),
            "narrative_1_id": self.narrative_1_id(),
            "narrative_2_id": self.narrative_2_id(),
            "anecdotal_in_relationship_ids": self.anecdotal_in_relationship_ids(),
        }

    @staticmethod
    def create():
        temporal_relationship = TemporalEventRelationship()
        temporal_relationship.set_id(str(uuid.uuid4()))
        return temporal_relationship
