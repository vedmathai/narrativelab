from typing import List
import uuid

from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class ActorRelationship(AbstractRelationship):
    _type = "actor_relationship"

    def __init__(self):
        super().__init__()
        self._narrative_id = None
        self._actor_id = None

    def narrative_id(self) -> str:
        return self._narrative_id

    def narrative(self):
        return self._narrative_graph.id2narrative_node(self._narrative_id)

    def actor_id(self) -> str:
        return self._actor_id

    def actor(self) -> str:
        return self._narrative_graph.id2entity_node(self._actor_id)

    def set_narrative(self, narrative):
        self._narrative_id = narrative.id()

    def set_actor(self, actor):
        self._actor_id = actor.id()

    def set_narrative_id(self, narrative_id: str):
        self._narrative_id = narrative_id

    def set_actor_id(self, actor_id: str):
        self._actor_id = actor_id

    def nodes(self):
        self._nodes = [
            self.actor(),
            self.narrative()
        ]
        return super().nodes()

    @staticmethod
    def from_dict(val, narrative_graph):
        actor_relationship = ActorRelationship()
        actor_relationship.set_narrative_graph(narrative_graph)
        actor_relationship.set_narrative_id(val['narrative_id'])
        actor_relationship.set_actor_id(val['actor_id'])
        return actor_relationship

    def to_dict(self):
        return {
            "id": self.id(),
            'narrative_id': self.narrative_id(),
            'actor_id': self.actor_id(),
        }

    @staticmethod
    def create():
        actor_relationship = ActorRelationship()
        actor_relationship.set_id(str(uuid.uuid4()))
        return actor_relationship
