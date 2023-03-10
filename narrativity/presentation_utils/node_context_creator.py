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
            'entity': [
                self.get_narratives,
            ],
            'narrative': [
                self.get_actors,
            ]
        }

    
    def create(self, node):
        node_context = NodeContext()
        node_context.set_node(node)
        self.get_actors(node, node_context)
        return node_context
        
    def get_actors(self, node, node_context: NodeContext):
        for actor in node.actors():
            node_context.add_id2node(actor.id(), actor)
            node_context.add_key('actors')
            node_context.add_key2id('actors', actor.id())
