from typing import Dict, List


from narrativity.datamodel.narrative_graph.nodes.abstract_node import AbstractNode
from narrativity.datamodel.narrative_graph.auxiliaries.temporal_value import TemporalValue


class AbsoluteTemporalNode(AbstractNode):
    _type = "absolute_temporal_node"

    def __init__(self):
        self._id = ""
        self._unit = None
        self._temporal_values = []
        self._narrative_relationship_ids = []
        
    def id(self) -> str:
        return self._id

    def temporal_values(self) -> List[TemporalValue]:
        return self._temporal_values

    def narrative_relationship_ids(self):
        return self._narrative_relationship_ids

    def narrative_relationships(self):
        return [self._narrative_graph.id2action_relationships(id) for id in self.narrative_relationship_ids()]

    def set_id(self, id: str) -> None:
        self._id = id

    def set_unit(self, unit: str) -> None:
        self._unit = unit

    def set_temporal_values(self, temporal_values: List[TemporalValue]) -> None:
        self._temporal_values = temporal_values

    def set_narrative_relationship_ids(self, narrative_relationships_ids: List[str]):
        self._narrative_relationship_ids = narrative_relationships_ids

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
            "unit": self.unit(),
            "values": [i.to_dict() for i in self.values()],
            "narrative_relationship_ids": self.narrative_relationship_ids(),
        }
