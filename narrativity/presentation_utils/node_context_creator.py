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

    def add_other_node(self, node, key, node_context: NodeContext):
        node_context.add_id2node(node.id(), node)
        node_context.add_key(key)
        node_context.add_key2id(key, node.id())
        
    def get_actors(self, node, node_context: NodeContext):
        for actor_relationship in node.actor_relationships():
            actor = actor_relationship.actor()
            self.add_other_node(actor, 'actor', node_context)

    def get_direct_objects(self, node, node_context: NodeContext):
        for direct_object_relationship in node.direct_object_relationships():
            object = direct_object_relationship.object()
            self.add_other_node(object, 'direct_object', node_context)

    def get_indirect_objects(self, node, node_context: NodeContext):
        for indirect_object_relationship in node.indirect_object_relationships():
            object = indirect_object_relationship.object()
            self.add_other_node(object, 'indirect_object', node_context)

    def get_narratives(self, node, node_context: NodeContext):
        for narrative_relationship in node.narrative_relationships():
            narrative = narrative_relationship.narrative()
            self.add_other_node(narrative, 'narrative', node_context)
