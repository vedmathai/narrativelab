from typing import List

from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class ObjectRelationship(AbstractRelationship):
    def __init__(self):
        self._preposition = None
        self._narrative_id = None
        self._object_id = None

    def preposition(self) -> str:
        return self._preposition

    def narrative_id(self) -> str:
        return self._narrative_id

    def narrative(self):
        return self._narrative_graph.id2narrative_node(self._narrative_id)

    def object_id(self) -> str:
        return self._object_id

    def object(self) -> str:
        return self._narrative_graph.id2object_node(self._object_id)

    def set_preposition(self, preposition: str):
        self._preposition = preposition

    def set_narrative_id(self, narrative_id: str):
        self._narrative_id = narrative_id

    def set_object_id(self, object_id: str):
        self._object_id = object_id

    @staticmethod
    def from_dict(val, narrative_graph):
        object_relationship = ObjectRelationship()
        object_relationship.set_narrative_graph(narrative_graph)
        object_relationship.set_preposition(val['preposition'])
        object_relationship.set_narrative_id(val['narrative_id'])
        object_relationship.set_object_id(val['object_id'])
        return object_relationship

    def to_dict(self):
        return {
            "id": self.id(),
            'preposition': self.preposition(),
            'narrative_id': self.narrative_id(),
            'object_id': self.object_id(),
        }
