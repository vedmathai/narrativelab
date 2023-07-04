from typing import List
import uuid

from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship


class SubjectRelationship(AbstractRelationship):
    _type = "subject_relationship"

    def __init__(self):
        super().__init__()
        self._narrative_id = None
        self._subject_id = None

    def narrative_id(self) -> str:
        return self._narrative_id

    def narrative(self):
        return self._narrative_graph.id2narrative_node(self._narrative_id)

    def subject_id(self) -> str:
        return self._subject_id

    def subject(self) -> str:
        return self._narrative_graph.id2entity_node(self._subject_id)

    def set_narrative(self, narrative):
        self._narrative_id = narrative.id()

    def set_subject(self, subject):
        self._subject_id = subject.id()

    def set_narrative_id(self, narrative_id: str):
        self._narrative_id = narrative_id

    def set_subject_id(self, subject_id: str):
        self._subject_id = subject_id

    def nodes(self):
        self._nodes = [
            self.narrative(),
            self.subject(),
        ]
        return super().nodes()

    @staticmethod
    def from_dict(val, narrative_graph):
        subject_relationship = SubjectRelationship()
        subject_relationship.set_narrative_graph(narrative_graph)
        subject_relationship.set_narrative_id(val['narrative_id'])
        subject_relationship.set_subject_id(val['subject_id'])
        return subject_relationship

    def to_dict(self):
        return {
            "id": self.id(),
            'narrative_id': self.narrative_id(),
            'subject_id': self.subject_id(),
        }

    @staticmethod
    def create():
        subject_relationship = SubjectRelationship()
        subject_relationship.set_id(str(uuid.uuid4()))
        return subject_relationship
