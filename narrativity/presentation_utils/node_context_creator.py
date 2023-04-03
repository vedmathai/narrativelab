from narrativity.datamodel.node_context.node_context import NodeContext

class NodeContextCreator:

    _instance = None
    _name = "Node Context Creator"

    @classmethod
    def instantiate(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            raise Exception('{} not instantiated.'.format(cls._name))
        return cls._instance

    def load(self):
        pass

    def node_type2fns(self, node):
        node_type2fn = {
            'entity_node': [
                self.get_narratives,
            ],
            'narrative_node': [
                self.get_actors,
                self.get_direct_objects,
                self.get_indirect_objects,
                self.get_locations,
                self.get_absolute_times,
                self.get_subjects,
                self.get_states_descriptors,
                self.get_causal_out_relationships,
                self.get_causal_in_relationships,
                self.get_contradictory_out_relationships,
                self.get_contradictory_in_relationships,
                self.get_temporal_event_out_relationships,
                self.get_temporal_event_in_relationships,
            ],
            'absolute_temporal_node': [
                self.get_narratives,
            ]
        }
        return node_type2fn.get(node.type())

    def create(self, node):
        node_context = NodeContext()
        node_context.set_node(node)
        fns = self.node_type2fns(node)
        for fn in fns:
            fn(node, node_context)
        return node_context

    def add_other_node(self, other_node, key, node_context: NodeContext):
        node_context.add_id2node(other_node)
        node_context.add_key(key)
        node_context.add_key2id(key, other_node.id())

    def add_relationship(self, other_node, relationship, node_context: NodeContext):
        node_context.add_id2relationship(relationship)
        node_context.add_node_id2relationship_id(other_node, relationship)
        
    def get_actors(self, node, node_context: NodeContext):
        for actor_relationship in node.actor_relationships():
            actor = actor_relationship.actor()
            self.add_other_node(actor, 'actor', node_context)
            self.add_relationship(actor, actor_relationship, node_context)

    def get_direct_objects(self, node, node_context: NodeContext):
        for direct_object_relationship in node.direct_object_relationships():
            object = direct_object_relationship.object()
            self.add_other_node(object, 'direct_object', node_context)
            self.add_relationship(object, direct_object_relationship, node_context)

    def get_indirect_objects(self, node, node_context: NodeContext):
        for indirect_object_relationship in node.indirect_object_relationships():
            object = indirect_object_relationship.object()
            self.add_other_node(object, 'indirect_object', node_context)
            self.add_relationship(object, indirect_object_relationship, node_context)

    def get_locations(self, node, node_context: NodeContext):
        for location_relationship in node.location_relationships():
            location = location_relationship.location()
            self.add_other_node(location, 'location', node_context)
            self.add_relationship(location, location_relationship, node_context)

    def get_absolute_times(self, node, node_context: NodeContext):
        for absolute_temporal_relationship in node.absolute_temporal_relationships():
            absolute_time = absolute_temporal_relationship.absolute_temporal_node()
            self.add_other_node(absolute_time, 'absolute_time', node_context)
            self.add_relationship(absolute_time, absolute_temporal_relationship, node_context)

    def get_narratives(self, node, node_context: NodeContext):
        for narrative_relationship in node.narrative_relationships():
            narrative = narrative_relationship.narrative()
            self.add_other_node(narrative, 'narrative', node_context)
            self.add_relationship(narrative, narrative_relationship, node_context)

    def get_subjects(self, node, node_context: NodeContext):
        for subject_relationship in node.subject_relationships():
            subject = subject_relationship.subject()
            self.add_other_node(subject, 'subject', node_context)
            self.add_relationship(subject, subject_relationship, node_context)

    def get_states_descriptors(self, node, node_context: NodeContext):
        for state_relationship in node.state_relationships():
            state = state_relationship.state()
            self.add_other_node(state, 'state_descriptor', node_context)
            self.add_relationship(state, state_relationship, node_context)

    def get_causal_in_relationships(self, node, node_context: NodeContext):
        for causal_in_relationship in node.causal_in_relationships():
            narrative_1 = causal_in_relationship.narrative_1()
            self.add_other_node(narrative_1, 'causal_in', node_context)
            self.add_relationship(narrative_1, causal_in_relationship, node_context)

    def get_causal_out_relationships(self, node, node_context: NodeContext):
        for causal_out_relationship in node.causal_out_relationships():
            narrative_2 = causal_out_relationship.narrative_2()
            self.add_other_node(narrative_2, 'causal_out', node_context)
            self.add_relationship(narrative_2, causal_out_relationship, node_context)

    def get_contradictory_in_relationships(self, node, node_context: NodeContext):
        for contradictory_in_relationship in node.contradictory_in_relationships():
            narrative_1 = contradictory_in_relationship.narrative_1()
            self.add_other_node(narrative_1, 'contradictory_in', node_context)
            self.add_relationship(narrative_1, contradictory_in_relationship, node_context)

    def get_contradictory_out_relationships(self, node, node_context: NodeContext):
        for contradictory_out_relationship in node.contradictory_out_relationships():
            narrative_2 = contradictory_out_relationship.narrative_2()
            self.add_other_node(narrative_2, 'contradictory_out', node_context)
            self.add_relationship(narrative_2, contradictory_out_relationship, node_context)

    def get_temporal_event_in_relationships(self, node, node_context: NodeContext):
        for temporal_event_in_relationship in node.temporal_event_in_relationships():
            narrative_1 = temporal_event_in_relationship.narrative_1()
            self.add_other_node(narrative_1, 'temporal_event_in', node_context)
            self.add_relationship(narrative_1, temporal_event_in_relationship, node_context)

    def get_temporal_event_out_relationships(self, node, node_context: NodeContext):
        for temporal_event_out_relationship in node.temporal_event_out_relationships():
            narrative_2 = temporal_event_out_relationship.narrative_2()
            self.add_other_node(narrative_2, 'temporal_event_out', node_context)
            self.add_relationship(narrative_2, temporal_event_out_relationship, node_context)
