from typing import List
import uuid

from narrativity.datamodel.narrative_graph.nodes.abstract_node import AbstractNode


class NarrativeNode(AbstractNode):
    _type = "narrative_node"

    def __init__(self):
        self._canonical_name = None
        self._names = []
        self._actor_relationship_ids = []
        self._subject_relationship_ids = []
        self._action_ids = []
        self._direct_object_relationship_ids = []
        self._indirect_object_relationship_ids = [] #adjunct vs instrumental
        self._location_relationship_ids = []
        self._temporal_event_in_relationship_ids = []
        self._temporal_event_out_relationship_ids = []
        self._absolute_temporal_relationship_ids = []
        self._state_relationship_ids = []
        self._sub_narrative_ids = []
        self._parent_narrative_ids = []
        self._causal_out_relationship_ids = []
        self._causal_in_relationship_ids = []
        self._contradictory_out_relationship_ids = []
        self._contradictory_in_relationship_ids = []
        self._anecdotal_in_relationship_ids = []
        self._anecdotal_out_relationship_ids = []
        self._and_like_relationship_ids = []
        self._descriptor_relationship_ids = []
        self._cooccurrence_relationship_ids = []
        self._prep_in_relationship_ids = []
        self._prep_out_relationship_ids = []
        self._sources = []
        self._is_leaf = False
        self._is_state = False
        self._is_negative = False
        self._narrative_graph = None
        self._token = None

    def canonical_name(self) -> str:
        return self._canonical_name
        
    def names(self) -> List[str]:
        return self._names

    def display_name(self) -> str:
        if self._canonical_name is not None:
            return self.canonical_name()
        elif self.is_state() is False:
            actors = [i.actor() for i in self.actor_relationships()]
            actors = '|'.join(actor.display_name() for actor in actors)
            actions = '|'.join(action.display_name() for action in self.actions())
            if self.is_negative() is True:
                actions = 'not ( {} )'.format(actions)
            direct_objects = '|'.join(i.object().display_name() for i in self.direct_object_relationships())
            return f'{actors}->{actions}->{direct_objects}'
        elif self.is_state() is True:
            subjects = [i.subject() for i in self.subject_relationships()]
            subjects = '|'.join(subject.display_name() for subject in subjects)
            auxiliaries = '|'.join(i.auxiliary() for i in self.state_relationships())
            states = '|'.join(i.state().display_name() for i in self.state_relationships())
            return f'{subjects}->{auxiliaries}->{states}'

    def actor_relationship_ids(self) -> List[str]:
        return self._actor_relationship_ids

    def subject_relationship_ids(self) -> List[str]:
        return self._subject_relationship_ids

    def actor_relationships(self):
        return [self._narrative_graph.id2actor_relationship(i) for i in self.actor_relationship_ids()]

    def subject_relationships(self):
        return [self._narrative_graph.id2subject_relationship(i) for i in self.subject_relationship_ids()]

    def actions(self):
        return [self._narrative_graph.id2action_node(i) for i in self.action_ids()]

    def action_ids(self) -> List[str]:
        return self._action_ids

    def direct_object_relationships(self):
        return [self._narrative_graph.id2direct_object_relationship(i) for i in self.direct_object_relationship_ids()]

    def direct_object_relationship_ids(self) -> List[str]:
        return self._direct_object_relationship_ids

    def indirect_object_relationships(self):
        return [self._narrative_graph.id2indirect_object_relationship(i) for i in self.indirect_object_relationship_ids()]

    def indirect_object_relationship_ids(self) -> List[str]:
        return self._indirect_object_relationship_ids

    def location_relationship_ids(self) -> List[str]:
        return self._location_relationship_ids

    def location_relationships(self):
        return [self._narrative_graph.id2location_relationship(i) for i in self.location_relationship_ids()]

    def temporal_event_in_relationship_ids(self) -> List[str]:
        return self._temporal_event_in_relationship_ids

    def temporal_event_out_relationship_ids(self) -> List[str]:
        return self._temporal_event_out_relationship_ids

    def absolute_temporal_relationship_ids(self) -> List[str]:
        return self._absolute_temporal_relationship_ids

    def state_relationships(self):
        return [self._narrative_graph.id2state_relationship(i) for i in self.state_relationship_ids()]

    def state_relationship_ids(self) -> List[str]:
        return self._state_relationship_ids

    def temporal_event_in_relationships(self):
        return [self._narrative_graph.id2temporal_event_relationship(i) for i in self.temporal_event_in_relationship_ids()]

    def temporal_event_out_relationships(self):
        return [self._narrative_graph.id2temporal_event_relationship(i) for i in self.temporal_event_out_relationship_ids()]

    def absolute_temporal_relationships(self):
        return [self._narrative_graph.id2absolute_temporal_relationship(i) for i in self.absolute_temporal_relationship_ids()]

    def causal_in_relationship_ids(self):
        return self._causal_in_relationship_ids

    def causal_out_relationship_ids(self):
        return self._causal_out_relationship_ids

    def causal_in_relationships(self):
        return [self._narrative_graph.id2causal_relationship(i) for i in self.causal_in_relationship_ids()]

    def causal_out_relationships(self):
        return [self._narrative_graph.id2causal_relationship(i) for i in self.causal_out_relationship_ids()]

    def contradictory_in_relationship_ids(self):
        return self._contradictory_in_relationship_ids

    def contradictory_out_relationship_ids(self):
        return self._contradictory_out_relationship_ids

    def contradictory_in_relationships(self):
        return [self._narrative_graph.id2contradictory_relationship(i) for i in self.contradictory_in_relationship_ids()]

    def contradictory_out_relationships(self):
        return [self._narrative_graph.id2contradictory_relationship(i) for i in self.contradictory_out_relationship_ids()]

    def anecdotal_in_relationship_ids(self):
        return self._anecdotal_in_relationship_ids

    def anecdotal_out_relationship_ids(self):
        return self._anecdotal_out_relationship_ids

    def anecdotal_in_relationships(self):
        return [self._narrative_graph.id2anecdotal_relationship(i) for i in self.anecdotal_in_relationship_ids()]

    def anecdotal_out_relationships(self):
        return [self._narrative_graph.id2anecdotal_relationship(i) for i in self.anecdotal_out_relationship_ids()]
    
    def prep_in_relationship_ids(self):
        return self._prep_in_relationship_ids

    def prep_out_relationship_ids(self):
        return self._prep_out_relationship_ids

    def prep_in_relationships(self):
        return [self._narrative_graph.id2prep_relationship(i) for i in self.prep_in_relationship_ids()]

    def prep_out_relationships(self):
        return [self._narrative_graph.id2prep_relationship(i) for i in self.prep_out_relationship_ids()]
    
    def and_like_relationship_ids(self):
        return self._and_like_relationship_ids

    def and_like_relationships(self):
        return [self._narrative_graph.id2and_like_relationship(i) for i in self.and_like_relationship_ids()]
    
    def descriptor_relationship_ids(self):
        return self._descriptor_relationship_ids

    def descriptor_relationships(self):
        return [self._narrative_graph.id2descriptor_relationship(i) for i in self.descriptor_relationship_ids()]
    
    def cooccurrence_relationship_ids(self):
        return self._cooccurrence_relationship_ids

    def cooccurrence_relationships(self):
        return [self._narrative_graph.id2cooccurrence_relationship(i) for i in self.cooccurrence_relationship_ids()]

    def sub_narrative_ids(self) -> List[str]:
        return self._sub_narrative_ids

    def parent_narrative_ids(self) -> List[str]:
        return self._parent_narrative_ids

    def sub_narratives(self) -> List["NarrativeNode"]:
        return [self._narrative_graph.id2narrative_node(id) for i in self.sub_narrative_ids()]

    def parent_narratives(self) -> List["NarrativeNode"]:
        return [self._narrative_graph.id2narrative_node(id) for i in self.parent_narrative_ids()]

    def sources(self) -> List[str]:
        return self._sources

    def is_state(self) -> bool:
        return self._is_state

    def is_leaf(self) -> bool:
        return self._is_leaf
    
    def is_negative(self) -> bool:
        return self._is_negative
    
    def set_names(self, names: List[str]) -> None:
        self._names = names

    def set_actor_ids(self, actor_ids: List[str]) -> None:
        self._actor_ids = actor_ids

    def set_subject_ids(self, subject_ids: List[str]) -> None:
        self._subject_ids = subject_ids

    def add_actor_relationship(self, actor_relationship) -> None:
        self._actor_relationship_ids.append(actor_relationship.id())

    def add_subject_relationship(self, subject_relationship) -> None:
        self._subject_relationship_ids.append(subject_relationship.id())

    def set_action_ids(self, action_ids: List[str]) -> None:
        self._action_ids = action_ids

    def add_action(self, action) -> None:
        self._action_ids.append(action.id())

    def set_actor_relationship_ids(self, actor_relationship_id) -> List[str]:
        self._actor_relationship_ids = actor_relationship_id

    def set_direct_object_relationship_ids(self, direct_object_relationship_id) -> List[str]:
        self._direct_object_relationship_ids = direct_object_relationship_id

    def add_direct_object_relationship(self, direct_object_relationship) -> List[str]:
        self._direct_object_relationship_ids.append(direct_object_relationship.id())

    def set_indirect_object_relationship_ids(self, indirect_object_relationship_ids) -> List[str]:
        self._indirect_object_relationship_ids = indirect_object_relationship_ids

    def add_indirect_object_relationship(self, indirect_object_relationship) -> List[str]:
        self._indirect_object_relationship_ids.append(indirect_object_relationship.id())

    def set_location_relationship_ids(self, location_relationship_ids) -> List[str]:
        self._location_relationship_ids = location_relationship_ids

    def set_subject_relationship_ids(self, subject_relationship_ids) -> List[str]:
        self._subject_relationship_ids = subject_relationship_ids

    def add_location_relationship(self, location_relationship) -> List[str]:
        self._location_relationship_ids.append(location_relationship.id())

    def set_temporal_event_in_relationship_ids(self, temporal_event_in_relationship_ids) -> List[str]:
        self._temporal_event_in_relationship_ids = temporal_event_in_relationship_ids

    def set_temporal_event_out_relationship_ids(self, temporal_event_out_relationship_ids) -> List[str]:
        self._temporal_event_out_relationship_ids = temporal_event_out_relationship_ids

    def set_absolute_temporal_relationship_ids(self, absolute_temporal_relationship_ids) -> List[str]:
        self._absolute_temporal_relationship_ids = absolute_temporal_relationship_ids

    def set_state_relationship_ids(self, state_relationship_ids) -> List[str]:
        self._state_relationship_ids = state_relationship_ids

    def add_state_relationship(self, state_relationship) -> None:
        self._state_relationship_ids.append(state_relationship.id())

    def set_causal_in_relationship_ids(self, causal_in_relationship_ids) -> None:
        self._causal_in_relationship_ids = causal_in_relationship_ids

    def set_causal_out_relationship_ids(self, causal_out_relationship_ids) -> None:
        self._causal_out_relationship_ids = causal_out_relationship_ids

    def add_causal_in_relationship(self, causal_in_relationship) -> None:
        self._causal_in_relationship_ids.append(causal_in_relationship.id())

    def add_causal_out_relationship(self, causal_out_relationship) -> None:
        self._causal_out_relationship_ids.append(causal_out_relationship.id())

    def set_contradictory_in_relationship_ids(self, contradictory_in_relationship_ids) -> None:
        self._contradictory_in_relationship_ids = contradictory_in_relationship_ids

    def set_contradictory_out_relationship_ids(self, contradictory_out_relationship_ids) -> None:
        self._contradictory_out_relationship_ids = contradictory_out_relationship_ids

    def add_contradictory_in_relationship(self, contradictory_in_relationship) -> None:
        self._contradictory_in_relationship_ids.append(contradictory_in_relationship.id())

    def add_contradictory_out_relationship(self, contradictory_out_relationship) -> None:
        self._contradictory_out_relationship_ids.append(contradictory_out_relationship.id())

    def set_anecdotal_in_relationship_ids(self, anecdotal_in_relationship_ids) -> None:
        self._anecdotal_in_relationship_ids = anecdotal_in_relationship_ids

    def set_anecdotal_out_relationship_ids(self, anecdotal_out_relationship_ids) -> None:
        self._anecdotal_out_relationship_ids = anecdotal_out_relationship_ids

    def add_anecdotal_in_relationship(self, anecdotal_in_relationship) -> None:
        self._anecdotal_in_relationship_ids.append(anecdotal_in_relationship.id())

    def add_anecdotal_out_relationship(self, anecdotal_out_relationship) -> None:
        self._anecdotal_out_relationship_ids.append(anecdotal_out_relationship.id())

    def set_prep_in_relationship_ids(self, prep_in_relationship_ids) -> None:
        self._prep_in_relationship_ids = prep_in_relationship_ids

    def set_prep_out_relationship_ids(self, prep_out_relationship_ids) -> None:
        self._prep_out_relationship_ids = prep_out_relationship_ids

    def add_prep_in_relationship(self, prep_in_relationship) -> None:
        self._prep_in_relationship_ids.append(prep_in_relationship.id())

    def add_prep_out_relationship(self, prep_out_relationship) -> None:
        self._prep_out_relationship_ids.append(prep_out_relationship.id())

    def set_and_like_relationship_ids(self, and_like_relationship_ids) -> None:
        self._and_like_relationship_ids = and_like_relationship_ids

    def add_and_like_relationship(self, and_like_relationship) -> None:
        self._and_like_relationship_ids.append(and_like_relationship.id())

    def set_descriptor_relationship_ids(self, descriptor_relationship_ids) -> None:
        self._descriptor_relationship_ids = descriptor_relationship_ids

    def add_descriptor_relationship(self, descriptor_relationship) -> None:
        self._descriptor_relationship_ids.append(descriptor_relationship.id())

    def set_cooccurrence_relationship_ids(self, cooccurrence_relationship_ids) -> None:
        self._cooccurrence_relationship_ids = cooccurrence_relationship_ids

    def add_cooccurrence_relationship(self, cooccurrence_relationship) -> None:
        self._cooccurrence_relationship_ids.append(cooccurrence_relationship.id())

    def add_temporal_event_in_relationship(self, temporal_event_in_relationship) -> None:
        self._temporal_event_in_relationship_ids.append(temporal_event_in_relationship.id())

    def add_temporal_event_out_relationship(self, temporal_event_out_relationship) -> None:
        self._temporal_event_out_relationship_ids.append(temporal_event_out_relationship.id())

    def add_absolute_temporal_relationship(self, absolute_temporal_relationship) -> None:
        self._absolute_temporal_relationship_ids.append(absolute_temporal_relationship.id())

    def set_sub_narrative_ids(self, sub_narrative_ids) -> List[str]:
        self._sub_narrative_ids = sub_narrative_ids

    def set_parent_narrative_ids(self, parent_narrative_ids: List[str]) -> None:
        self._parent_narrative_ids = parent_narrative_ids

    def set_sources(self, sources: List[str]) -> None:
        self._sources = sources

    def set_is_state(self, is_state: bool) -> None:
        self._is_state = is_state

    def set_is_leaf(self, is_leaf: bool) -> None:
        self._is_leaf = is_leaf

    def set_is_negative(self, is_negative: bool) -> None:
        self._is_negative = is_negative

    def set_canonical_name(self, canonical_name: str) -> None:
        self._canonical_name = canonical_name

    def relationships(self):
        self._relationships = [
            self.actor_relationships(),
            self.subject_relationships(),
            self.actions(),
            self.direct_object_relationships(),
            self.indirect_object_relationships(),
            self.location_relationships(),
            self.temporal_event_in_relationships(),
            self.temporal_event_out_relationships(),
            self.absolute_temporal_relationships(),
            self.state_relationships(),
            self.causal_in_relationships(),
            self.causal_out_relationships(),
            self.contradictory_in_relationships(),
            self.contradictory_out_relationships(),
            self.anecdotal_in_relationships(),
            self.anecdotal_out_relationships(),
            self.prep_in_relationships(),
            self.prep_out_relationships(),
            self.and_like_relationships(),
            self.descriptor_relationships(),
            self.cooccurrence_relationships(),
        ]
        return super().relationships()

    @staticmethod
    def from_dict(val, narrative_graph):
        narrative_node = NarrativeNode()
        narrative_node.set_id(val['id'])
        narrative_node.set_narrative_graph(narrative_graph)
        narrative_node.set_canonical_name(val['canonical_name'])
        narrative_node.set_names(val['names'])
        narrative_node.set_actor_relationship_ids(val['actor_relationship_ids'])
        narrative_node.set_subject_relationship_ids(val['subject_relationship_ids'])
        narrative_node.set_action_ids(val['action_ids'])
        narrative_node.set_direct_object_relationship_ids(val['direct_object_relationship_ids'])
        narrative_node.set_indirect_object_relationship_ids(val['indirect_object_relationship_ids'])
        narrative_node.set_location_relationship_ids(val['location_relationship_ids'])
        narrative_node.set_temporal_event_in_relationship_ids(val['temporal_event_in_relationship_ids'])
        narrative_node.set_temporal_event_out_relationship_ids(val['temporal_event_out_relationship_ids'])
        narrative_node.set_absolute_temporal_relationship_ids(val['absolute_temporal_relationship_ids'])
        narrative_node.set_state_relationship_ids(val['state_relationship_ids'])
        narrative_node.set_causal_in_relationship_ids(val['causal_in_relationship_ids'])
        narrative_node.set_causal_out_relationship_ids(val['causal_out_relationship_ids'])
        narrative_node.set_contradictory_in_relationship_ids(val['contradictory_in_relationship_ids'])
        narrative_node.set_contradictory_out_relationship_ids(val['contradictory_out_relationship_ids'])
        narrative_node.set_anecdotal_in_relationship_ids(val['anecdotal_in_relationship_ids'])
        narrative_node.set_anecdotal_out_relationship_ids(val['anecdotal_out_relationship_ids'])
        narrative_node.set_prep_in_relationship_ids(val['prep_in_relationship_ids'])
        narrative_node.set_prep_out_relationship_ids(val['prep_out_relationship_ids'])
        narrative_node.set_and_like_relationship_ids(val['and_like_relationship_ids'])
        narrative_node.set_descriptor_relationship_ids(val['descriptor_relationship_ids'])
        narrative_node.set_cooccurrence_relationship_ids(val['cooccurrence_relationship_ids'])
        narrative_node.set_sub_narrative_ids(val['sub_narrative_ids'])
        narrative_node.set_parent_narrative_ids(val['parent_narrative_ids'])
        narrative_node.set_sources(val['sources'])
        narrative_node.set_is_state(val['is_state'])
        narrative_node.set_is_negative(val['is_negative'])
        narrative_node.set_canonical_name(val['canonical_name'])
        return narrative_node

    def to_dict(self):
        return {
            "id": self.id(),
            "canonical_name": self.canonical_name(),
            "display_name": self.display_name(),
            "names": self.names(),
            "actor_relationship_ids": self.actor_relationship_ids(),
            "subject_relationship_ids": self.subject_relationship_ids(),
            "action_ids": self.action_ids(),
            "direct_object_relationship_ids": self.direct_object_relationship_ids(),
            "indirect_object_relationship_ids": self.indirect_object_relationship_ids(),
            "location_relationship_ids": self.location_relationship_ids(),
            "temporal_event_in_relationship_ids": self.temporal_event_in_relationship_ids(),
            "temporal_event_out_relationship_ids": self.temporal_event_out_relationship_ids(),
            "absolute_temporal_relationship_ids": self.absolute_temporal_relationship_ids(),
            "state_relationship_ids": self.state_relationship_ids(),
            "causal_in_relationship_ids": self.causal_in_relationship_ids(),
            "causal_out_relationship_ids": self.causal_out_relationship_ids(),
            "contradictory_in_relationship_ids": self.contradictory_in_relationship_ids(),
            "contradictory_out_relationship_ids": self.contradictory_out_relationship_ids(),
            "anecdotal_in_relationship_ids": self.anecdotal_in_relationship_ids(),
            "anecdotal_out_relationship_ids": self.anecdotal_out_relationship_ids(),
            "prep_in_relationship_ids": self.prep_in_relationship_ids(),
            "prep_out_relationship_ids": self.prep_out_relationship_ids(),
            "and_like_relationship_ids": self.and_like_relationship_ids(),
            "descriptor_relationship_ids": self.descriptor_relationship_ids(),
            "cooccurrence_relationship_ids": self.cooccurrence_relationship_ids(),
            "sub_narrative_ids": self.sub_narrative_ids(),
            "parent_narrative_ids": self.parent_narrative_ids(),
            "sources": self.sources(),
            "is_state": self.is_state(),
            "is_negative": self.is_negative(),
            "canonical_name": self.canonical_name(),
        }

    @staticmethod
    def create():
        narrative_node = NarrativeNode()
        narrative_node.set_id(str(uuid.uuid4()))
        return narrative_node
