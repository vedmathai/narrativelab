from typing import List
import uuid

from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class AndLikeRelationship(AbstractRelationship):
    _type = "and_like_relationship"

    def __init__(self):
        self._is_to_relationship = None
        self._narrative_1_id = None
        self._narrative_2_id = None
        self._relationship_id = None

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

    @staticmethod
    def from_dict(val, narrative_graph):
        and_like_relationship = AndLikeRelationship()
        and_like_relationship.set_id(val['id'])
        and_like_relationship.set_narrative_graph(narrative_graph)
        and_like_relationship.set_is_to_relationship(val['is_to_relationship'])
        and_like_relationship.set_relationship(val['relationship'])
        and_like_relationship.set_narrative_1_id(val['narrative_1_id'])
        and_like_relationship.set_narrative_2_id(val['narrative_2_id'])
        return and_like_relationship
 
    def to_dict(self):
        return {
            "id": self.id(),
            "is_to_relationship": self.is_to_relationship(),
            "relationship": self.relationship(),
            "narrative_1_id": self.narrative_1_id(),
            "narrative_2_id": self.narrative_2_id(),
        }

    @staticmethod
    def create():
        and_like_relationship = AndLikeRelationship()
        and_like_relationship.set_id(str(uuid.uuid4()))
        return and_like_relationship
