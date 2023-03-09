from typing import List, Dict
import uuid

from narrativity.datamodel.narrative_graph.nodes.abstract_node import AbstractNode


class EntityNode(AbstractNode):
    def __init__(self):
        super().__init__()
        self._canonical_name = ""
        self._narrative_relationship_ids = []
        
    def canonical_name(self) -> str:
        return self._canonical_name

    def set_canonical_name(self, canonical_name: str) -> None:
        self._canonical_name = canonical_name

    def narrative_relationship_ids(self):
        return self._narrative_relationship_ids

    def narrative_relationships(self):
        return [self._narrative_graph.id2entity_relationships(id) for id in self.narrative_relationship_ids()]

    def set_narrative_relationship_ids(self, narrative_relationships_ids: List[str]):
        self._narrative_relationship_ids = narrative_relationships_ids

    def add_narrative_relationship(self, narrative_relationship):
        self._narrative_relationship_ids.append(narrative_relationship.id())

    @staticmethod
    def from_dict(val, narrative_graph):
        entity_node = EntityNode()
        entity_node.set_id(val['id'])
        entity_node.set_narrative_graph(narrative_graph)
        entity_node.set_canonical_name(val['canonical_name'])
        entity_node.set_narrative_relationship_ids(val['narrative_relationship_ids'])
        return entity_node

    def to_dict(self) -> Dict:
        return {
            "id": self.id(),
            "canonical_name": self.canonical_name(),
            "narrative_relationship_ids": self.narrative_relationship_ids(),
        }

    @staticmethod
    def create():
        entity_node = EntityNode()
        entity_node.set_id(str(uuid.uuid4()))
        return entity_node
