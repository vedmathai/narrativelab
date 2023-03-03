from typing import Dict, List

from narrativity.datamodel.narrative_graph.nodes.abstract_node import AbstractNode


class ActionNode(AbstractNode):
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
        return [self._narrative_graph.id2action_relationships(id) for id in self.narrative_relationship_ids()]

    def set_narrative_relationship_ids(self, narrative_relationships_ids: List[str]):
        self._narrative_relationship_ids = narrative_relationships_ids

    @staticmethod
    def from_dict(val, narrative_graph):
        action_node = ActionNode()
        action_node.set_id(val['id'])
        action_node.set_narrative_graph(narrative_graph)
        action_node.set_canonical_name(val['canonical_name'])
        action_node.set_narrative_relationship_ids(val['narrative_relationship_ids'])
        return action_node

    def to_dict(self) -> Dict:
        return {
            "id": self.id(),
            "canonical_name": self.canonical_name(),
            "narrative_relationship_ids": self.narrative_relationship_ids(),
        }
