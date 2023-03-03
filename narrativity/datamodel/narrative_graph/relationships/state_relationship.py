from typing import List


from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class StateRelationship(AbstractRelationship):
    def __init__(self):
        self._auxiliary = None
        self._narrative_id = None
        self._state_id = None

    def auxiliary(self) -> str:
        return self._auxiliary
    
    def narrative_id(self) -> str:
        return self._narrative_id

    def narrative(self) -> str:
        return self._narrative_graph.id2narrative_node(self._narrative_id)

    def state_id(self) -> str:
        return self._state_id

    def state(self) -> str:
        return self._narrative_graph.id2state_node(self._state_id)

    def set_narrative_id(self, narrative_id: str):
        self._narrative_id = narrative_id

    def set_auxiliary(self, auxiliary: str):
        self._auxiliary = auxiliary

    def set_state_id(self, state_id: str):
        self._state_id = state_id

    @staticmethod
    def from_dict(val, narrative_graph):
        state_relationship = StateRelationship()
        state_relationship.set_id(val['id'])
        state_relationship.set_narrative_graph(narrative_graph)
        state_relationship.set_narrative_id(val['narrative_id'])
        state_relationship.set_auxiliary(val['auxiliary'])
        state_relationship.set_state_id(val['state_id'])
        return state_relationship

    def to_dict(self):
        return {
            "id": self.id(),
            'narrative': self.narrative_id(),
            'auxiliary': self.auxiliary(),
            'state': self.state_id(),
        }
