import uuid

from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class ContradictoryRelationship(AbstractRelationship):
    _type = "contradictory_relationship"

    def __init__(self):
        super().__init__()
        self._narrative_1_id = None
        self._narrative_2_id = None
        self._mark = None

    def narrative_1_id(self) -> str:
        return self._narrative_1_id
 
    def narrative_2_id(self) -> str:
        return self._narrative_2_id

    def narrative_1(self):
        return self._narrative_graph.id2narrative_node(self._narrative_1_id)

    def narrative_2(self):
        return self._narrative_graph.id2narrative_node(self._narrative_2_id)

    def mark(self) -> str:
        return self._mark

    def set_narrative_1(self, narrative):
        self._narrative_1_id = narrative.id()

    def set_narrative_2(self, narrative):
        self._narrative_2_id = narrative.id()

    def set_mark(self, mark):
        self._mark = mark

    def set_narrative_1_id(self, narrative_id: str):
        self._narrative_1_id = narrative_id

    def set_narrative_2_id(self, narrative_id: str):
        self._narrative_2_id = narrative_id

    def set_actor_id(self, actor_id: str):
        self._actor_id = actor_id

    @staticmethod
    def from_dict(val, narrative_graph):
        causal_relationship = ContradictoryRelationship()
        causal_relationship.set_narrative_graph(narrative_graph)
        causal_relationship.set_narrative_1_id(val['narrative_1_id'])
        causal_relationship.set_narrative_2_id(val['narrative_2_id'])
        causal_relationship.set_mark(val['mark'])
        return causal_relationship

    def to_dict(self):
        return {
            "narrative_1_id": self.narrative_1_id(),
            "narrative_2_id": self.narrative_2_id(),
            "mark": self.mark(),
        }

    @staticmethod
    def create():
        causal_relationship = ContradictoryRelationship()
        causal_relationship.set_id(str(uuid.uuid4()))
        return causal_relationship
