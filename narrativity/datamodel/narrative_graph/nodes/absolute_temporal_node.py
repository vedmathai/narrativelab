from typing import Dict, List
import uuid

from narrativity.datamodel.narrative_graph.nodes.abstract_node import AbstractNode
from narrativity.datamodel.narrative_graph.auxiliaries.temporal_value import TemporalValue


class AbsoluteTemporalNode(AbstractNode):
    _type = "absolute_temporal_node"

    def __init__(self):
        super().__init__()
        self._temporal_values = []
        self._narrative_relationship_ids = []

    def canonical_name(self) -> str:
        name = '_'.join(i.value() for i in self._temporal_values)
        return name

    def display_name(self) -> str:
        return self.canonical_name()
        
    def temporal_values(self) -> List[TemporalValue]:
        return self._temporal_values

    def narrative_relationship_ids(self):
        return self._narrative_relationship_ids

    def narrative_relationships(self):
        return [self._narrative_graph.id2absolute_temporal_relationship(id) for id in self.narrative_relationship_ids()]

    def add_temporal_value(self, temporal_value: TemporalValue):
        self._temporal_values.append(temporal_value)

    def set_id(self, id: str) -> None:
        self._id = id

    def set_temporal_values(self, temporal_values: List[TemporalValue]) -> None:
        self._temporal_values = temporal_values

    def set_narrative_relationship_ids(self, narrative_relationships_ids: List[str]):
        self._narrative_relationship_ids = narrative_relationships_ids

    def add_narrative_relationship(self, narrative_relationship):
        self._narrative_relationship_ids.append(narrative_relationship.id())

    def relationships(self):
        self._relationships = [
            self.narrative_relationships()
        ]
        return super().relationships()

    @staticmethod
    def from_dict(val, narrative_graph) -> "AbsoluteTemporalNode":
        absolute_temporal_node = AbsoluteTemporalNode
        absolute_temporal_node.set_id(val['id'])
        absolute_temporal_node.set_narrative_graph(narrative_graph)
        absolute_temporal_node.set_temporal_values(TemporalValue.from_dict(i) for i in val['temporal_values'])
        absolute_temporal_node.set_narrative_relationship_ids(val['narrative_relationship_ids'])
        return absolute_temporal_node

    def to_dict(self) -> Dict:
        return {
            "id": self.id(),
            "display_name": self.display_name(),
            "temporal_values": [i.to_dict() for i in self.temporal_values()],
            "narrative_relationship_ids": self.narrative_relationship_ids(),
        }

    @staticmethod
    def create():
        action_node = AbsoluteTemporalNode()
        action_node.set_id(str(uuid.uuid4()))
        return action_node
