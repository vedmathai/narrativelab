from typing import List
import uuid


from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class LocationRelationship(AbstractRelationship):
    _type = "location_relationship"

    def __init__(self):
        self._preposition = None
        self._narrative = None
        self._location = None

    def preposition(self) -> str:
        return self._preposition

    def narrative_id(self) -> str:
        return self._narrative_id

    def narrative(self) -> str:
        return self._narrative_graph.id2narrative_node(self._narrative_id)

    def location_id(self) -> str:
        return self._location_id

    def location(self):
        return self._narrative_graph.id2entity_node(self._location_id)

    def set_narrative(self, narrative):
        self._narrative_id = narrative.id()

    def set_preposition(self, preposition: str):
        self._preposition = preposition

    def set_narrative_id(self, narrative_id: str):
        self._narrative_id = narrative_id

    def set_location(self, location):
        self._location_id = location.id()

    def set_location_id(self, location_id: str):
        self._location_id = location_id

    def nodes(self):
        self._nodes = [
            self.narrative(),
            self.location(),
        ]
        return super().nodes()

    @staticmethod
    def from_dict(val, narrative_graph):
        location_relationship = LocationRelationship()
        location_relationship.set_narrative_graph(narrative_graph)
        location_relationship.set_preposition(val['preposition'])
        location_relationship.set_narrative_id(val['narrative_id'])
        location_relationship.set_location_id(val['location_id'])
        return location_relationship

    def to_dict(self):
        return {
            "id": self.id(),
            'preposition': self.preposition(),
            'narrative_id': self.narrative_id(),
            'location_id': self.location_id(),
        }

    @staticmethod
    def create():
        location_relationship = LocationRelationship()
        location_relationship.set_id(str(uuid.uuid4()))
        return location_relationship
