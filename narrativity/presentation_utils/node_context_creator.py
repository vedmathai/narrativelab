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

    def get_narratives(self, node, node_context: NodeContext):
        for narrative_relationship in node.narrative_relationships():
            narrative = narrative_relationship.narrative()
            self.add_other_node(narrative, 'narrative', node_context)
            self.add_relationship(narrative, narrative_relationship, node_context)
