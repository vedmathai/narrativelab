import uuid

from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class MainNarrativeRelationship(AbstractRelationship):
    _type = "main_narrative_relationship"

    def __init__(self):
        super().__init__()
        self._narrative_1_id = None
        self._narrative_2_id = None

    def display_name(self):
        narrative_1 = self.narrative_1().display_name()
        narrative_2 = self.narrative_2().display_name()
        main = "{} is the main narrative of {}".format(narrative_1, narrative_2)
        return main

    def narrative_1_id(self) -> str:
        return self._narrative_1_id
 
    def narrative_2_id(self) -> str:
        return self._narrative_2_id

    def narrative_1(self):
        return self._narrative_graph.id2narrative_node(self._narrative_1_id)

    def narrative_2(self):
        return self._narrative_graph.id2narrative_node(self._narrative_2_id)

    def set_narrative_1(self, narrative):
        self._narrative_1_id = narrative.id()

    def set_narrative_2(self, narrative):
        self._narrative_2_id = narrative.id()

    def set_narrative_1_id(self, narrative_id: str):
        self._narrative_1_id = narrative_id

    def set_narrative_2_id(self, narrative_id: str):
        self._narrative_2_id = narrative_id

    def nodes(self):
        self._nodes = [
            self.narrative_1(),
            self.narrative_2(),
        ]
        return super().nodes()

    @staticmethod
    def from_dict(val, narrative_graph):
        main_narrative_relationship = MainNarrativeRelationship()
        main_narrative_relationship.set_narrative_graph(narrative_graph)
        main_narrative_relationship.set_narrative_1_id(val['narrative_1_id'])
        main_narrative_relationship.set_narrative_2_id(val['narrative_2_id'])
        return main_narrative_relationship

    def to_dict(self):
        return {
            "id": self.id(),
            "display_name": self.display_name(),
            "narrative_1_id": self.narrative_1_id(),
            "narrative_2_id": self.narrative_2_id(),
        }

    @staticmethod
    def create():
        main_narrative_relationship = MainNarrativeRelationship()
        main_narrative_relationship.set_id(str(uuid.uuid4()))
        return main_narrative_relationship
