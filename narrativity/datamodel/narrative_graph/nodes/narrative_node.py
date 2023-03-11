from typing import List
import uuid

from narrativity.datamodel.narrative_graph.nodes.abstract_node import AbstractNode


class NarrativeNode(AbstractNode):
    _type = "narrative_node"

    def __init__(self):
        self._canonical_name = None
        self._names = []
        self._actor_ids = []
        self._action_ids = []
        self._direct_object_relationship_ids = []
        self._indirect_object_relationship_ids = [] #adjunct vs instrumental
        self._location_relationship_ids = []
        self._temporal_relationship_ids = []
        self._state_relationship_ids = []
        self._sub_narrative_ids = []
        self._parent_narrative_ids = []
        self._sources = []
        self._state_ids = []
        self._is_leaf = False
        self._is_state = False
        self._narrative_graph = None

    def canonical_name(self) -> str:
        return self._canonical_name
        
    def names(self) -> List[str]:
        return self._names

    def actors(self):
        return [self._narrative_graph.id2entity_node(i) for i in self.actor_ids()]

    def actor_ids(self) -> List[str]:
        return self._actor_ids

    def actions(self):
        return [self._narrative_graph.id2action_node(i) for i in self.action_ids()]

    def action_ids(self) -> List[str]:
        return self._action_ids

    def direct_object_relationships(self):
        return [self._narrative_graph.id2entity_node(i) for i in self.direct_object_relationship_ids()]

    def direct_object_relationship_ids(self) -> List[str]:
        return self._direct_object_relationship_ids

    def indirect_object_relationships(self):
        return [self._narrative_graph.id2entity_node(i) for i in self.indirect_object_relationship_ids()]

    def indirect_object_relationship_ids(self) -> List[str]:
        return self._indirect_object_relationship_ids

    def location_relationship_ids(self) -> List[str]:
        return self._location_relationship_ids

    def location_relationships(self):
        return [self._narrative_graph.id2location_relationship(i) for i in self.location_relationship_ids()]

    def temporal_relationship_ids(self) -> List[str]:
        return self._temporal_relationship_ids

    def state_relationship_ids(self) -> List[str]:
        return self._state_relationship_ids

    def temporal_relationships(self):
        return [self._narrative_graph.id2temporal_relationship(i) for i in self.temporal_relationship_ids()]

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

    def states(self) -> List[str]:
        return [self._narrative_graph.id2state_node(id) for i in self.state_ids()]
 
    def state_ids(self) -> List[str]:
        return self._state_ids

    def is_state(self) -> bool:
        return self._is_state

    def is_leaf(self) -> bool:
        return self._is_leaf

    def set_names(self, names: List[str]) -> None:
        self._names = names

    def set_actor_ids(self, actor_ids: List[str]) -> None:
        self._actor_ids = actor_ids

    def add_actor(self, actor) -> None:
        self._actor_ids.append(actor.id())

    def set_action_ids(self, action_ids: List[str]) -> None:
        self._action_ids = action_ids

    def add_action(self, action) -> None:
        self._action_ids.append(action.id())

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

    def set_temporal_relationship_ids(self, temporal_relationship_ids) -> List[str]:
        self._temporal_relationship_ids = temporal_relationship_ids

    def set_state_relationship_ids(self, state_relationship_ids) -> List[str]:
        self._state_relationship_ids = state_relationship_ids

    def add_state_relationship(self, state_relationship) -> None:
        self._state_relationship_ids.append(state_relationship.id())

    def set_sub_narrative_ids(self, sub_narrative_ids) -> List[str]:
        self._sub_narrative_ids = sub_narrative_ids

    def set_parent_narrative_ids(self, parent_narrative_ids: List[str]) -> None:
        self._parent_narrative_ids = parent_narrative_ids

    def set_sources(self, sources: List[str]) -> None:
        self._sources = sources

    def set_state_ids(self, state_ids: List[str]) -> None:
        self._state_ids = state_ids

    def set_is_state(self, is_state: bool) -> None:
        self._is_state = is_state

    def set_is_leaf(self, is_leaf: bool) -> None:
        self._is_leaf = is_leaf

    def set_canonical_name(self, canonical_name: str) -> None:
        self._canonical_name = canonical_name

    @staticmethod
    def from_dict(val, narrative_graph):
        narrative_node = NarrativeNode()
        narrative_node.set_id(val['id'])
        narrative_node.set_narrative_graph(narrative_graph)
        narrative_node.set_canonical_name(val['canonical_name'])
        narrative_node.set_names(val['names'])
        narrative_node.set_actor_ids(val['actor_ids'])
        narrative_node.set_action_ids(val['action_ids'])
        narrative_node.set_direct_object_relationship_ids(val['direct_object_relationship_ids'])
        narrative_node.set_indirect_object_relationship_ids(val['indirect_object_relationship_ids'])
        narrative_node.set_location_relationship_ids(val['location_relationship_ids'])
        narrative_node.set_temporal_relationship_ids(val['temporal_relationship_ids'])
        narrative_node.set_state_relationship_ids(val['state_relationship_ids'])
        narrative_node.set_sub_narrative_ids(val['sub_narrative_ids'])
        narrative_node.set_parent_narrative_ids(val['parent_narrative_ids'])
        narrative_node.set_sources(val['sources'])
        narrative_node.set_state_ids(val['state_ids'])
        narrative_node.set_is_state(val['is_state'])
        narrative_node.set_canonical_name(val['canonical_name'])
        return narrative_node

    def to_dict(self):
        return {
            "id": self.id(),
            "canonical_name": self.canonical_name(),
            "names": self.names(),
            "actor_ids": self.actor_ids(),
            "action_ids": self.action_ids(),
            "direct_object_relationship_ids": self.direct_object_relationship_ids(),
            "indirect_object_relationship_ids": self.indirect_object_relationship_ids(),
            "location_relationship_ids": self.location_relationship_ids(),
            "temporal_relationship_ids": self.temporal_relationship_ids(),
            "state_relationship_ids": self.state_relationship_ids(),
            "sub_narrative_ids": self.sub_narrative_ids(),
            "parent_narrative_ids": self.parent_narrative_ids(),
            "sources": self.sources(),
            "state_ids": self.state_ids(),
            "is_state": self.is_state(),
            "canonical_name": self.canonical_name(),
        }

    @staticmethod
    def create():
        narrative_node = NarrativeNode()
        narrative_node.set_id(str(uuid.uuid4()))
        return narrative_node
