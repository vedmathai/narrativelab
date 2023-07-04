from typing import List
import uuid

from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class DescriptorRelationship(AbstractRelationship):
    _type = "descriptor_relationship"

    def __init__(self):
        self._entity_node_id = None
        self._narrative_id = None

    def display_name(self):
        entity_node = self.entity_node().display_name()
        narrative = self.narrative().display_name()
        descriptor_relationship = "{} that {}".format(entity_node, narrative)
        return descriptor_relationship

    def entity_node(self):
        return self._narrative_graph.id2entity_node(self._entity_node_id)

    def narrative(self):
        return self._narrative_graph.id2narrative_node(self._narrative_id)

    def narrative_id(self):
        return self._narrative_id

    def entity_node_id(self):
        return self._entity_node_id

    def set_entity_node(self, entity_node: List[str]) -> None:
        self._entity_node_id = entity_node.id()

    def set_entity_node_id(self, entity_node_id: str) -> None:
        self._entity_node_id = entity_node_id

    def set_narrative_id(self, narrative_id: str) -> None:
        self._narrative_id = narrative_id

    def set_narrative(self, narrative: str) -> None:
        self._narrative_id = narrative.id()

    def nodes(self):
        self._nodes = [
            self.entity_node(),
            self.narrative(),
        ]
        return super().nodes()

    @staticmethod
    def from_dict(val, narrative_graph):
        descriptor_relationship = DescriptorRelationship()
        descriptor_relationship.set_id(val['id'])
        descriptor_relationship.set_narrative_graph(narrative_graph)
        descriptor_relationship.set_narrative_id(val['narrative_id'])
        descriptor_relationship.set_entity_node_id(val['entity_node_id'])
        return descriptor_relationship
 
    def to_dict(self):
        return {
            "id": self.id(),
            "display_name": self.display_name(),
            "narrative_id": self.narrative_id(),
            "entity_node_id": self.entity_node_id(),
        }

    @staticmethod
    def create():
        descriptor_relationship = DescriptorRelationship()
        descriptor_relationship.set_id(str(uuid.uuid4()))
        return descriptor_relationship
