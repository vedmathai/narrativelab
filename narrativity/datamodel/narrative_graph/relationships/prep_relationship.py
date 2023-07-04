from typing import List
import uuid

from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class PrepRelationship(AbstractRelationship):
    _type = "prep_relationship"

    def __init__(self):
        self._is_to_relationship = None
        self._narrative_1_id = None
        self._narrative_2_id = None
        self._relationship_id = None
        self._preposition = None

    def display_name(self) -> str:
        return self._preposition

    def preposition(self) -> str:
        return self._preposition

    def is_to_relationship(self):
        return self._is_to_relationship

    def narrative_1_id(self):
        return self._narrative_1_id

    def narrative_2_id(self):
        return self._narrative_2_id

    def narrative_1(self):
        return self._narrative_graph.id2narrative_node(self._narrative_1_id)

    def narrative_2(self):
        return self._narrative_graph.id2narrative_node(self._narrative_2_id)

    def relationship_id(self):
        return self._relationship_id

    def relationship(self):
        return self._narrative_graph.id2relationship(self._relationship_id)

    def set_is_to_relationship(self, is_to_relationship):
        self._is_to_relationship = is_to_relationship

    def set_relationship_id(self, relationship_id: str) -> None:
        self._relationship_id = relationship_id

    def set_narrative_1_id(self, narrative_1_id: str) -> None:
        self._narrative_1_id = narrative_1_id

    def set_narrative_2_id(self, narrative_2_id: str) -> None:
        self._narrative_2_id = narrative_2_id

    def set_narrative_1(self, narrative_1) -> None:
        self._narrative_1_id = narrative_1.id()

    def set_narrative_2(self, narrative_2) -> None:
        self._narrative_2_id = narrative_2.id()

    def set_relationship(self, relationship) -> None:
        self._relationship_id = relationship.id()

    def set_preposition(self, preposition: str):
        self._preposition = preposition

    def nodes(self):
        self._nodes = [
            self.narrative_1(),
            self.narrative_2(),
            self.relationship(),
        ]
        return super().nodes()

    @staticmethod
    def from_dict(val, narrative_graph):
        prep_relationship = PrepRelationship()
        prep_relationship.set_id(val['id'])
        prep_relationship.set_narrative_graph(narrative_graph)
        prep_relationship.set_is_to_relationship(val['is_to_relationship'])
        prep_relationship.set_relationship(val['relationship'])
        prep_relationship.set_narrative_1_id(val['narrative_1_id'])
        prep_relationship.set_narrative_2_id(val['narrative_2_id'])
        prep_relationship.set(val['narrative_2_id'])
        prep_relationship.set_preposition(val['preposition'])
        return prep_relationship
 
    def to_dict(self):
        return {
            "id": self.id(),
            "is_to_relationship": self.is_to_relationship(),
            "relationship": self.relationship(),
            "narrative_1_id": self.narrative_1_id(),
            "narrative_2_id": self.narrative_2_id(),
            "preposition": self.preposition(),
        }

    @staticmethod
    def create():
        prep_relationship = PrepRelationship()
        prep_relationship.set_id(str(uuid.uuid4()))
        return prep_relationship
