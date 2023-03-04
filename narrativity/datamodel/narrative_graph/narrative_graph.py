from typing import Dict

from narrativity.datamodel.narrative_graph.nodes.abstract_node import AbstractNode
from narrativity.datamodel.narrative_graph.relationships.abstract_relationship import AbstractRelationship
from narrativity.datamodel.narrative_graph.nodes.action_node import ActionNode
from narrativity.datamodel.narrative_graph.nodes.entity_node import EntityNode
from narrativity.datamodel.narrative_graph.nodes.absolute_temporal_node import AbsoluteTemporalNode
from narrativity.datamodel.narrative_graph.nodes.narrative_node import NarrativeNode
from narrativity.datamodel.narrative_graph.nodes.state_node import StateNode
from narrativity.datamodel.narrative_graph.relationships.location_relationship import LocationRelationship
from narrativity.datamodel.narrative_graph.relationships.object_relationship import ObjectRelationship
from narrativity.datamodel.narrative_graph.relationships.temporal_relationship import TemporalRelationship
from narrativity.datamodel.narrative_graph.relationships.state_relationship import StateRelationship



class NarrativeGraph:
    def __init__(self):
        self._action_nodes: Dict[str, ActionNode] = {}
        self._entity_nodes: Dict[str, EntityNode]= {}
        self._absolute_temporal_nodes: Dict[str, AbsoluteTemporalNode] = {}
        self._narrative_nodes: Dict[str, NarrativeNode] = {}
        self._state_nodes: Dict[str, StateNode] = {}
        self._location_relationships: Dict[str, LocationRelationship] = {}
        self._object_relationships: Dict[str, ObjectRelationship] = {}
        self._temporal_relationships: Dict[str, TemporalRelationship] = {}
        self._state_relationships: Dict[str, StateRelationship] = {}

    def action_nodes(self) -> Dict[str, ActionNode]:
        return self._action_nodes

    def entity_nodes(self) -> Dict[str, EntityNode]:
        return self._entity_nodes

    def absolute_temporal_nodes(self) -> Dict[str, AbsoluteTemporalNode]:
        return self._absolute_temporal_nodes

    def narrative_nodes(self) -> Dict[str, NarrativeNode]:
        return self._narrative_nodes

    def id2action_node(self, id: str) -> ActionNode:
        return self._action_nodes.get(id)

    def id2entity_node(self, id: str) -> EntityNode:
        return self._entity_nodes.get(id)

    def id2absolute_temporal_node(self, id: str) -> AbsoluteTemporalNode:
        return self._absolute_temporal_nodes.get(id)

    def id2narrative_node(self, id: str) -> NarrativeNode:
        return self._narrative_nodes.get(id)

    def id2state_node(self, id: str) -> StateNode:
        return self._state_nodes.get(id)

    def location_relationships(self) -> Dict[str, LocationRelationship]:
        return self._location_relationships

    def object_relationships(self) -> Dict[str, ObjectRelationship]:
        return self._object_relationships

    def temporal_relationships(self) -> Dict[str, TemporalRelationship]:
        return self._temporal_relationships

    def state_relationships(self) -> Dict[str, StateRelationship]:
        return self._state_relationships

    def id2location_relationship(self, id: str) -> LocationRelationship:
        return self._location_relationships.get(id)

    def id2object_relationships(self, id: str) -> ObjectRelationship:
        return self._object_relationships.get(id)

    def id2temporal_relationship(self, id: str) -> TemporalRelationship:
        return self._temporal_relationships.get(id)

    def id2state_relationship(self, id: str) -> StateRelationship:
        return self._state_relationships.get(id)

    def id2node(self, id: str) -> AbstractNode:
        id2nodefns = [
            self.id2action_node,
            self.id2entity_node,
            self.id2absolute_temporal_node,
            self.id2narrative_node,
            self.id2state_node,
        ]
        for fn in id2nodefns:
            node = fn(id)
            if node is not None:
                return node
        return None

    def id2relationship(self, id: str) -> AbstractRelationship:
        id2relationshipfns = [
            self.id2location_relationship,
            self.id2object_relationships,
            self.id2temporal_relationship,
            self.id2state_relationship,
        ]
        for fn in id2relationshipfns:
            relationship = fn(id)
            if relationship is not None:
                return relationship
        return None

    def set_action_nodes(self, action_nodes: Dict[str, ActionNode]) -> None:
        self._action_nodes = action_nodes

    def set_entity_nodes(self, entity_nodes: Dict[str, EntityNode]) -> None:
        self._entity_nodes = entity_nodes

    def set_absolute_temporal_nodes(self, absolute_temporal_nodes: Dict[str, AbsoluteTemporalNode]) -> None:
        self._absolute_temporal_nodes = absolute_temporal_nodes
    
    def set_narrative_nodes(self, narrative_nodes: Dict[str, NarrativeNode]) -> None:
        self._narrative_nodes = narrative_nodes

    def set_state_nodes(self, state_nodes: Dict[str, StateNode]) -> None:
        self._state_nodes = state_nodes

    def set_location_relationships(self, location_relationships: Dict[str, LocationRelationship]) -> None:
        self._location_relationships = location_relationships

    def set_temporal_relationships(self, temporal_relationships: Dict[str, TemporalRelationship]) -> None:
        self._temporal_relationships = temporal_relationships
    
    def set_object_relationships(self, object_relationships: Dict[str, ObjectRelationship]) -> None:
        self._object_relationships = object_relationships

    def set_state_relationships(self, state_relationships: Dict[str, StateRelationship]) -> None:
        self._state_relationships = state_relationships

    def add_action_node(self, action_node: ActionNode) -> None:
        self._action_nodes[action_node.id()] = action_node

    def add_entity_nodes(self, entity_node: EntityNode) -> None:
        self._entity_nodes[entity_node.id()] = entity_node

    def add_state_nodes(self, state_node: StateNode) -> None:
        self._state_nodes[state_node.id()] = state_node

    def add_absolute_temporal_nodes(self, absolute_temporal_node: AbsoluteTemporalNode) -> None:
        self._absolute_temporal_nodes[absolute_temporal_node.id()] = absolute_temporal_node
    
    def add_narrative_nodes(self, narrative_node: NarrativeNode) -> None:
        self._narrative_node[narrative_node.id()] = narrative_node

    def add_location_relationship(self, location_relationship: LocationRelationship) -> None:
        self._location_relationships[location_relationship.id()] = location_relationship

    def add_temporal_relationships(self, temporal_relationship: TemporalRelationship) -> None:
        self._temporal_relationships[temporal_relationship.id()] = temporal_relationship
    
    def add_object_relationship(self, object_relationship: ObjectRelationship) -> None:
        self._object_relationships[object_relationship.id()] = object_relationship

    def add_state_relationship(self, state_relationship: StateRelationship) -> None:
        self._state_relationships[state_relationship.id()] = state_relationship

    def to_dict(self):
        return {
            "action_nodes": [i.to_dict() for i in self.action_nodes().values()],
            "entity_nodes": [i.to_dict() for i in self.entity_nodes().values()],
            "absolute_temporal_nodes": [i.to_dict() for i in self.absolute_temporal_nodes().values()],
            "narrative_nodes": [i.to_dict() for i in self.narrative_nodes().values()],
            "state_nodes": [i.to_dict() for i in self.state_nodes().values()],
            "location_relationships": [i.to_dict() for i in self.location_relationships().values()],
            "object_relationships": [i.to_dict() for i in self.object_relationships().values()],
            "temporal_relationships": [i.to_dict() for i in self.temporal_relationships().values()],
            "state_relationships": [i.to_dict() for i in self.state_relationships().values()],
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
        narrative_graph.set_state_nodes({
            i['id']: StateNode.from_dict(i) for i in val['state_nodes']
        })
        narrative_graph.set_location_relationships({
            i['id']: LocationRelationship.from_dict(i) for i in val['location_relationships']
        })
        narrative_graph.set_object_relationships({
            i['id']: ObjectRelationship.from_dict(i) for i in val['object_relationships']
        })
        narrative_graph.set_location_relationships({
            i['id']: LocationRelationship.from_dict(i) for i in val['location_relationships']
        })
        narrative_graph.set_state_relationships({
            i['id']: StateRelationship.from_dict(i) for i in val['state_relationships']
        })