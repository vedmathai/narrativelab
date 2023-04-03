import uuid

from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class CausalRelationship(AbstractRelationship):
    _type = "causal_relationship"

    def __init__(self):
        super().__init__()
        self._narrative_1_id = None
        self._narrative_2_id = None
        self._anecdotal_in_relationship_ids = []
        self._mark = None

    def display_name(self):
        narrative_1 = self.narrative_1().display_name()
        narrative_2 = self.narrative_2().display_name()
        causal = "{} therefore {}".format(narrative_1, narrative_2)
        return causal

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

    def anecdotal_in_relationship_ids(self):
        return self._anecdotal_in_relationship_ids
    
    def anecdotal_in_relationships(self):
        return [self._narrative_graph.id2anecdotal_relationship(i) for i in self._anecdotal_relationship_ids]

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

    def add_anecdotal_in_relationship_id(self, anecdotal_in_relationship_id):
        self._anecdotal_in_relationship_ids.append(anecdotal_in_relationship_id)

    def add_anecdotal_in_relationship(self, anecdotal_in_relationship):
        self._anecdotal_in_relationship_ids.append(anecdotal_in_relationship.id())

    def set_anecdotal_in_relationship_ids(self, anecdotal_in_relationship_ids):
        self._anecdotal_in_relationship_ids = anecdotal_in_relationship_ids

    @staticmethod
    def from_dict(val, narrative_graph):
        causal_relationship = CausalRelationship()
        causal_relationship.set_narrative_graph(narrative_graph)
        causal_relationship.set_narrative_1_id(val['narrative_1_id'])
        causal_relationship.set_narrative_2_id(val['narrative_2_id'])
        causal_relationship.set_anecdotal_in_relationship_ids(val['anecdotal_in_relationship_ids'])
        causal_relationship.set_mark(val['mark'])
        return causal_relationship

    def to_dict(self):
        return {
            "id": self.id(),
            "display_name": self.display_name(),
            "narrative_1_id": self.narrative_1_id(),
            "narrative_2_id": self.narrative_2_id(),
            "anecdotal_in_relationship_ids": self.anecdotal_in_relationship_ids(),
            "mark": self.mark(),
        }

    @staticmethod
    def create():
        causal_relationship = CausalRelationship()
        causal_relationship.set_id(str(uuid.uuid4()))
        return causal_relationship
