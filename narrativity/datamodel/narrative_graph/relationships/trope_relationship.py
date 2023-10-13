from typing import List
import uuid

from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class TropeRelationship(AbstractRelationship):
    _type = "trope_relationship"

    def __init__(self):
        super().__init__()
        self._narrative_id = None
        self._trope_id = None

    def narrative_id(self) -> str:
        return self._narrative_id

    def narrative(self):
        return self._narrative_graph.id2narrative_node(self._narrative_id)

    def trope_id(self) -> str:
        return self._trope_id

    def trope(self) -> str:
        return self._narrative_graph.id2trope_node(self._trope_id)

    def set_narrative(self, narrative):
        self._narrative_id = narrative.id()

    def set_trope(self, trope):
        self._trope_id = trope.id()

    def set_narrative_id(self, narrative_id: str):
        self._narrative_id = narrative_id

    def set_trope_id(self, trope_id: str):
        self._trope_id = trope_id

    def nodes(self):
        self._nodes = [
            self.narrative(),
            self.trope(),
        ]
        return super().nodes()

    @staticmethod
    def from_dict(val, narrative_graph):
        trope_relationship = TropeRelationship()
        trope_relationship.set_narrative_graph(narrative_graph)
        trope_relationship.set_narrative_id(val['narrative_id'])
        trope_relationship.set_trope_id(val['trope_id'])
        return trope_relationship

    def to_dict(self):
        return {
            "id": self.id(),
            'narrative_id': self.narrative_id(),
            'trope_id': self.trope_id(),
        }

    @staticmethod
    def create():
        trope_relationship = TropeRelationship()
        trope_relationship.set_id(str(uuid.uuid4()))
        return trope_relationship
