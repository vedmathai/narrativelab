from typing import Dict, List

from narrativity.datamodel.narrative_graph.nodes.abstract_node import AbstractNode
from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship
from narrativity.datamodel.narrative_graph.nodes.action_node import ActionNode
from narrativity.datamodel.narrative_graph.nodes.entity_node import EntityNode
from narrativity.datamodel.narrative_graph.nodes.absolute_temporal_node import AbsoluteTemporalNode
from narrativity.datamodel.narrative_graph.nodes.narrative_node import NarrativeNode
from narrativity.datamodel.narrative_graph.relationships.location_relationship import LocationRelationship
from narrativity.datamodel.narrative_graph.relationships.object_relationship import ObjectRelationship
from narrativity.datamodel.narrative_graph.relationships.temporal_event_relationship import TemporalEventRelationship
from narrativity.datamodel.narrative_graph.relationships.absolute_temporal_relationship import AbsoluteTemporalRelationship
from narrativity.datamodel.narrative_graph.relationships.state_relationship import StateRelationship
from narrativity.datamodel.narrative_graph.relationships.actor_relationship import ActorRelationship
from narrativity.datamodel.narrative_graph.relationships.subject_relationship import SubjectRelationship
from narrativity.datamodel.narrative_graph.relationships.causal_relationship import CausalRelationship
from narrativity.datamodel.narrative_graph.relationships.contradictory_relationship import ContradictoryRelationship
from narrativity.datamodel.narrative_graph.relationships.anecdotal_relationship import AnecdotalRelationship
from narrativity.datamodel.narrative_graph.relationships.prep_relationship import PrepRelationship
from narrativity.datamodel.narrative_graph.relationships.and_like_relationship import AndLikeRelationship
from narrativity.datamodel.narrative_graph.relationships.descriptor_relationship import DescriptorRelationship
from narrativity.datamodel.narrative_graph.relationships.cooccurrence_relationship import CooccurrenceRelationship


class NarrativeGraph:
    def __init__(self):
        self._action_nodes: Dict[str, ActionNode] = {}
        self._entity_nodes: Dict[str, EntityNode] = {}
        self._absolute_temporal_nodes: Dict[str, AbsoluteTemporalNode] = {}
        self._narrative_nodes: Dict[str, NarrativeNode] = {}
        self._location_relationships: Dict[str, LocationRelationship] = {}
        self._actor_relationships: Dict[str, ActorRelationship] = {}
        self._direct_object_relationships: Dict[str, ObjectRelationship] = {}
        self._indirect_object_relationships: Dict[str, ObjectRelationship] = {}
        self._temporal_event_relationships: Dict[str, TemporalEventRelationship] = {}
        self._absolute_temporal_relationships: Dict[str, AbsoluteTemporalRelationship] = {}
        self._state_relationships: Dict[str, StateRelationship] = {}
        self._subject_relationships: Dict[str, SubjectRelationship] = {}
        self._causal_relationships: Dict[str, CausalRelationship] = {}
        self._contradictory_relationships: Dict[str, ContradictoryRelationship] = {}
        self._anecdotal_relationships: Dict[str, AnecdotalRelationship] = {}
        self._prep_relationships: Dict[str, PrepRelationship] = {}
        self._and_like_relationships: Dict[str, AndLikeRelationship] = {}
        self._descriptor_relationships: Dict[str, DescriptorRelationship] = {}
        self._cooccurrence_relationships: Dict[str, CooccurrenceRelationship] = {}
        self._text2action_node: Dict[str, ActionNode] = {}
        self._text2entity_node: Dict[str, EntityNode] = {}
        self._text2absolute_temporal_node: Dict[str, AbsoluteTemporalNode] = {}

    def action_nodes(self) -> Dict[str, ActionNode]:
        return self._action_nodes

    def entity_nodes(self) -> Dict[str, EntityNode]:
        return self._entity_nodes

    def absolute_temporal_nodes(self) -> Dict[str, AbsoluteTemporalNode]:
        return self._absolute_temporal_nodes

    def narrative_nodes(self) -> Dict[str, NarrativeNode]:
        return self._narrative_nodes

    def text2action_node(self, text: str) -> ActionNode:
        return self._text2action_node.get(str)

    def id2action_node(self, id: str) -> ActionNode:
        return self._action_nodes.get(id)

    def text2entity_node(self, text: str) -> EntityNode:
        return self._text2entity_node.get(text)

    def id2entity_node(self, id: str) -> EntityNode:
        return self._entity_nodes.get(id)

    def text2entity_node(self, text: str) -> EntityNode:
        return self._text2entity_node.get(text)

    def id2absolute_temporal_node(self, id: str) -> AbsoluteTemporalNode:
        return self._absolute_temporal_nodes.get(id)

    def text2absolute_temporal_node(self, text: str) -> AbsoluteTemporalNode:
        return self._text2absolute_temporal_node.get(text)

    def id2narrative_node(self, id: str) -> NarrativeNode:
        return self._narrative_nodes.get(id)

    def actor_relationships(self) -> Dict[str, ActorRelationship]:
        return self._actor_relationships

    def subject_relationships(self) -> Dict[str, SubjectRelationship]:
        return self._subject_relationships

    def location_relationships(self) -> Dict[str, LocationRelationship]:
        return self._location_relationships

    def direct_object_relationships(self) -> Dict[str, ObjectRelationship]:
        return self._direct_object_relationships

    def indirect_object_relationships(self) -> Dict[str, ObjectRelationship]:
        return self._indirect_object_relationships

    def temporal_event_relationships(self) -> Dict[str, TemporalEventRelationship]:
        return self._temporal_event_relationships

    def absolute_temporal_relationships(self) -> Dict[str, AbsoluteTemporalRelationship]:
        return self._absolute_temporal_relationships

    def state_relationships(self) -> Dict[str, StateRelationship]:
        return self._state_relationships

    def causal_relationships(self) -> Dict[str, CausalRelationship]:
        return self._causal_relationships

    def contradictory_relationships(self) -> Dict[str, ContradictoryRelationship]:
        return self._contradictory_relationships

    def anecdotal_relationships(self) -> Dict[str, AnecdotalRelationship]:
        return self._anecdotal_relationships
    
    def prep_relationships(self) -> Dict[str, PrepRelationship]:
        return self._prep_relationships
    
    def and_like_relationships(self) -> Dict[str, AndLikeRelationship]:
        return self._and_like_relationships

    def id2and_like_relationship(self, id) -> Dict[str, AndLikeRelationship]:
        return self._and_like_relationships.get(id)
    
    def descriptor_relationships(self) -> Dict[str, DescriptorRelationship]:
        return self._descriptor_relationships

    def id2descriptor_relationship(self, id) -> Dict[str, DescriptorRelationship]:
        return self._descriptor_relationships.get(id)
    
    def cooccurrence_relationships(self) -> Dict[str, CooccurrenceRelationship]:
        return self._cooccurrence_relationships

    def id2cooccurrence_relationship(self, id) -> Dict[str, CooccurrenceRelationship]:
        return self._cooccurrence_relationships.get(id)

    def id2actor_relationship(self, id: str) -> ActorRelationship:
        return self._actor_relationships.get(id)

    def id2subject_relationship(self, id: str) -> SubjectRelationship:
        return self._subject_relationships.get(id)

    def id2location_relationship(self, id: str) -> LocationRelationship:
        return self._location_relationships.get(id)

    def id2direct_object_relationship(self, id: str) -> ObjectRelationship:
        return self._direct_object_relationships.get(id)

    def id2indirect_object_relationship(self, id: str) -> ObjectRelationship:
        return self._indirect_object_relationships.get(id)

    def id2absolute_temporal_relationship(self, id: str) -> AbsoluteTemporalRelationship:
        return self._absolute_temporal_relationships.get(id)

    def id2temporal_event_relationship(self, id: str) -> TemporalEventRelationship:
        return self._temporal_event_relationships.get(id)

    def id2state_relationship(self, id: str) -> StateRelationship:
        return self._state_relationships.get(id)

    def id2causal_relationship(self, id: str) -> CausalRelationship:
        return self._causal_relationships.get(id)

    def id2contradictory_relationship(self, id: str) -> ContradictoryRelationship:
        return self._contradictory_relationships.get(id)

    def id2anecdotal_relationship(self, id: str) -> AnecdotalRelationship:
        return self._anecdotal_relationships.get(id)
    
    def id2prep_relationship(self, id: str) -> PrepRelationship:
        return self._prep_relationships.get(id)

    def id2node(self, id: str) -> AbstractNode:
        id2nodefns = [
            self.id2action_node,
            self.id2entity_node,
            self.id2absolute_temporal_node,
            self.id2narrative_node,
        ]
        for fn in id2nodefns:
            node = fn(id)
            if node is not None:
                return node
        return None

    def id2relationship(self, id: str) -> AbstractRelationship:
        id2relationshipfns = [
            self.id2location_relationship,
            self.id2direct_object_relationship,
            self.id2indirect_object_relationship,
            self.id2temporal_event_relationship,
            self.id2absolute_temporal_relationship,
            self.id2state_relationship,
            self.id2actor_relationship,
            self.id2subject_relationship,
            self.id2causal_relationship,
            self.id2contradictory_relationship,
            self.id2anecdotal_relationship,
            self.id2prep_relationship,
            self.id2and_like_relationship,
            self.id2descriptor_relationship,
            self.id2cooccurrence_relationship,
        ]
        for fn in id2relationshipfns:
            relationship = fn(id)
            if relationship is not None:
                return relationship
        return None

    def nodes(self) -> List:
        _nodes = [
            self.action_nodes(),
            self.entity_nodes(),
            self.absolute_temporal_nodes(),
            self.narrative_nodes(),
        ]
        return sum([list(i.values()) for i in _nodes], [])

    def relationships(self) -> List:
        _relationships = [
            self.location_relationships(),
            self.direct_object_relationships(),
            self.indirect_object_relationships(),
            self.temporal_event_relationships(),
            self.absolute_temporal_relationships(),
            self.state_relationships(),
            self.actor_relationships(),
            self.subject_relationships(),
            self.causal_relationships(),
            self.contradictory_relationships(),
            self.anecdotal_relationships(),
            self.prep_relationships(),
            self.and_like_relationships(),
            self.descriptor_relationships(),
            self.cooccurrence_relationships(),
        ]
        return sum([list(i.values()) for i in _relationships], [])

    def set_action_nodes(self, action_nodes: Dict[str, ActionNode]) -> None:
        self._action_nodes = action_nodes
        for action_node in self._action_nodes.values():
            if action_node.canonical_name() is not None:
                self._text2entity_node[action_node.canonical_name()] = action_node

    def set_entity_nodes(self, entity_nodes: Dict[str, EntityNode]) -> None:
        self._entity_nodes = entity_nodes
        for entity_node in self._entity_nodes.values():
            if entity_node.canonical_name() is not None:
                self._text2entity_node[entity_node.canonical_name()] = entity_node

    def set_absolute_temporal_nodes(self, absolute_temporal_nodes: Dict[str, AbsoluteTemporalNode]) -> None:
        self._absolute_temporal_nodes = absolute_temporal_nodes
        for absolute_temporal_node in self._absolute_temporal_nodes.values():
            if absolute_temporal_node.canonical_name() is not None:
                self._text2absolute_temporal_node[absolute_temporal_node.canonical_name()] = absolute_temporal_node

    def set_absolute_temporal_nodes(self, absolute_temporal_nodes: Dict[str, AbsoluteTemporalNode]) -> None:
        self._absolute_temporal_nodes = absolute_temporal_nodes

    def set_narrative_nodes(self, narrative_nodes: Dict[str, NarrativeNode]) -> None:
        self._narrative_nodes = narrative_nodes

    def set_location_relationships(self, location_relationships: Dict[str, LocationRelationship]) -> None:
        self._location_relationships = location_relationships

    def set_temporal_event_relationships(self, temporal_event_relationships: Dict[str, TemporalEventRelationship]) -> None:
        self._temporal_event_relationships = temporal_event_relationships
        
    def set_absolute_temporal_relationships(self, absolute_temporal_relationships: Dict[str, AbsoluteTemporalRelationship]) -> None:
        self._absolute_temporal_relationships = absolute_temporal_relationships

    def set_direct_object_relationships(self, direct_object_relationships: Dict[str, ObjectRelationship]) -> None:
        self._direct_object_relationships = direct_object_relationships

    def set_indirect_object_relationships(self, indirect_object_relationships: Dict[str, ObjectRelationship]) -> None:
        self._indirect_object_relationships = indirect_object_relationships

    def set_state_relationships(self, state_relationships: Dict[str, StateRelationship]) -> None:
        self._state_relationships = state_relationships

    def set_actor_relationships(self, actor_relationships: Dict[str, ActorRelationship]) -> None:
        self._actor_relationships = actor_relationships

    def set_subject_relationships(self, subject_relationships: Dict[str, SubjectRelationship]) -> None:
        self._subject_relationships = subject_relationships

    def set_causal_relationships(self, causal_relationships: Dict[str, CausalRelationship]) -> None:
        self._causal_relationships = causal_relationships

    def set_contradictory_relationships(self, contradictory_relationships: Dict[str, ContradictoryRelationship]) -> None:
        self._contradictory_relationships = contradictory_relationships

    def set_anecdotal_relationships(self, anecdotal_relationships: Dict[str, AnecdotalRelationship]) -> None:
        self._anecdotal_relationships = anecdotal_relationships

    def set_prep_relationships(self, prep_relationships: Dict[str, PrepRelationship]) -> None:
        self._prep_relationships = prep_relationships

    def set_and_like_relationships(self, and_like_relationships: Dict[str, AndLikeRelationship]) -> None:
        self._and_like_relationships = and_like_relationships

    def set_descriptor_relationships(self, descriptor_relationships: Dict[str, DescriptorRelationship]) -> None:
        self._descriptor_relationships = descriptor_relationships

    def set_cooccurrence_relationships(self, cooccurrence_relationships: Dict[str, CooccurrenceRelationship]) -> None:
        self._cooccurrence_relationships = cooccurrence_relationships

    def add_action_node(self, action_node: ActionNode) -> None:
        self._action_nodes[action_node.id()] = action_node
        self._text2action_node[action_node.canonical_name()] = action_node

    def add_entity_node(self, entity_node: EntityNode) -> None:
        self._entity_nodes[entity_node.id()] = entity_node
        self._text2entity_node[entity_node.canonical_name()] = entity_node

    def add_absolute_temporal_node(self, absolute_temporal_node: AbsoluteTemporalNode) -> None:
        self._absolute_temporal_nodes[absolute_temporal_node.id()] = absolute_temporal_node
        self._text2absolute_temporal_node[absolute_temporal_node.canonical_name()] = absolute_temporal_node

    def add_narrative_node(self, narrative_node: NarrativeNode) -> None:
        self._narrative_nodes[narrative_node.id()] = narrative_node

    def add_location_relationship(self, location_relationship: LocationRelationship) -> None:
        self._location_relationships[location_relationship.id()] = location_relationship

    def add_temporal_event_relationship(self, temporal_event_relationship: TemporalEventRelationship) -> None:
        self._temporal_event_relationships[temporal_event_relationship.id()] = temporal_event_relationship
    
    def add_absolute_temporal_relationship(self, absolute_temporal_relationship: AbsoluteTemporalRelationship) -> None:
        self._absolute_temporal_relationships[absolute_temporal_relationship.id()] = absolute_temporal_relationship

    def add_direct_object_relationship(self, object_relationship: ObjectRelationship) -> None:
        self._direct_object_relationships[object_relationship.id()] = object_relationship

    def add_indirect_object_relationship(self, object_relationship: ObjectRelationship) -> None:
        self._indirect_object_relationships[object_relationship.id()] = object_relationship

    def add_state_relationship(self, state_relationship: StateRelationship) -> None:
        self._state_relationships[state_relationship.id()] = state_relationship

    def add_actor_relationship(self, actor_relationship: ActorRelationship) -> None:
        self._actor_relationships[actor_relationship.id()] = actor_relationship

    def add_subject_relationship(self, subject_relationship: SubjectRelationship) -> None:
        self._subject_relationships[subject_relationship.id()] = subject_relationship

    def add_causal_relationship(self, causal_relationship: CausalRelationship) -> None:
        self._causal_relationships[causal_relationship.id()] = causal_relationship

    def add_contradictory_relationship(self, contradictory_relationship: ContradictoryRelationship) -> None:
        self._contradictory_relationships[contradictory_relationship.id()] = contradictory_relationship

    def add_anecdotal_relationship(self, anecdotal_relationship: AnecdotalRelationship) -> None:
        self._anecdotal_relationships[anecdotal_relationship.id()] = anecdotal_relationship

    def add_prep_relationship(self, prep_relationship: PrepRelationship) -> None:
        self._prep_relationships[prep_relationship.id()] = prep_relationship

    def add_and_like_relationship(self, and_like_relationship: AndLikeRelationship) -> None:
        self._and_like_relationships[and_like_relationship.id()] = and_like_relationship

    def add_descriptor_relationship(self, descriptor_relationship: DescriptorRelationship) -> None:
        self._descriptor_relationships[descriptor_relationship.id()] = descriptor_relationship

    def add_cooccurrence_relationship(self, cooccurrence_relationship: CooccurrenceRelationship) -> None:
        self._cooccurrence_relationships[cooccurrence_relationship.id()] = cooccurrence_relationship

    def to_dict(self):
        return {
            "action_nodes": [i.to_dict() for i in self.action_nodes().values()],
            "entity_nodes": [i.to_dict() for i in self.entity_nodes().values()],
            "absolute_temporal_nodes": [i.to_dict() for i in self.absolute_temporal_nodes().values()],
            "narrative_nodes": [i.to_dict() for i in self.narrative_nodes().values()],
            "location_relationships": [i.to_dict() for i in self.location_relationships().values()],
            "direct_object_relationships": [i.to_dict() for i in self.direct_object_relationships().values()],
            "indirect_object_relationships": [i.to_dict() for i in self.indirect_object_relationships().values()],
            "temporal_event_relationships": [i.to_dict() for i in self.temporal_event_relationships().values()],
            "absolute_temporal_relationships": [i.to_dict() for i in self.absolute_temporal_relationships().values()],
            "state_relationships": [i.to_dict() for i in self.state_relationships().values()],
            "actor_relationships": [i.to_dict() for i in self.actor_relationships().values()],
            "causal_relationships": [i.to_dict() for i in self.causal_relationships().values()],
            "contradictory_relationships": [i.to_dict() for i in self.contradictory_relationships().values()],
            "anecdotal_relationships": [i.to_dict() for i in self.anecdotal_relationships().values()],
            "prep_relationships": [i.to_dict() for i in self.prep_relationships().values()],
            "and_like_relationships": [i.to_dict() for i in self.and_like_relationships().values()],
            "descriptor_relationships": [i.to_dict() for i in self.descriptor_relationships().values()],
            "cooccurrence_relationships": [i.to_dict() for i in self.cooccurrence_relationships().values()],
            "subject_relationships": [i.to_dict() for i in self.subject_relationships().values()],
        }

    @staticmethod
    def from_dict(val):
        narrative_graph = NarrativeGraph()
        narrative_graph.set_action_nodes({
            i['id']: ActionNode.from_dict(i) for i in val['action_nodes']
        })
        narrative_graph.set_entity_nodes({
            i['id']: EntityNode.from_dict(i) for i in val['entity_nodes']
        })
        narrative_graph.set_absolute_temporal_nodes({
            i['id']: AbsoluteTemporalNode.from_dict(i) for i in val['absolute_temporal_nodes']
        })
        narrative_graph.set_narrative_nodes({
            i['id']: NarrativeNode.from_dict(i) for i in val['narrative_nodes']
        })
        narrative_graph.set_location_relationships({
            i['id']: LocationRelationship.from_dict(i) for i in val['location_relationships']
        })
        narrative_graph.set_direct_object_relationships({
            i['id']: ObjectRelationship.from_dict(i) for i in val['direct_object_relationships']
        })
        narrative_graph.set_indirect_object_relationships({
            i['id']: ObjectRelationship.from_dict(i) for i in val['indirect_object_relationships']
        })
        narrative_graph.set_location_relationships({
            i['id']: LocationRelationship.from_dict(i) for i in val['location_relationships']
        })
        narrative_graph.set_state_relationships({
            i['id']: StateRelationship.from_dict(i) for i in val['state_relationships']
        })
        narrative_graph.set_state_relationships({
            i['id']: StateRelationship.from_dict(i) for i in val['state_relationships']
        })
        narrative_graph.set_subject_relationships({
            i['id']: SubjectRelationship.from_dict(i) for i in val['subject_relationships']
        })
        narrative_graph.set_causal_relationships({
            i['id']: CausalRelationship.from_dict(i) for i in val['causal_relationships']
        })
        narrative_graph.set_contradictory_relationships({
            i['id']: ContradictoryRelationship.from_dict(i) for i in val['contradictory_relationships']
        })
        narrative_graph.set_anecdotal_relationships({
            i['id']: AnecdotalRelationship.from_dict(i) for i in val['anecdotal_relationships']
        })
        narrative_graph.set_prep_relationships({
            i['id']: PrepRelationship.from_dict(i) for i in val['prep_relationships']
        })
        narrative_graph.set_and_like_relationships({
            i['id']: AndLikeRelationship.from_dict(i) for i in val['and_like_relationships']
        })  
        narrative_graph.set_descriptor_relationships({
            i['id']: DescriptorRelationship.from_dict(i) for i in val['descriptor_relationships']
        })
        narrative_graph.set_cooccurrence_relationships({
            i['id']: CooccurrenceRelationship.from_dict(i) for i in val['cooccurrence_relationships']
        })         
        narrative_graph.set_temporal_event_relationships({
            i['id']: TemporalEventRelationship.from_dict(i) for i in val['temporal_event_relationships']
        })
        narrative_graph.set_absolute_temporal_relationships({
            i['id']: AbsoluteTemporalRelationship.from_dict(i) for i in val['absolute_temporal_relationships']
        })
