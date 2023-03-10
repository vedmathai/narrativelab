from narrativity.graph_generator.dependency_parse_pipeline.parser import NarrativeGraphGenerator
from narrativity.presentation_utils.node_context_creator import NodeContextCreator


class NarrativityServer:

    def __init__(self):
        self._corpus = None

    _instance = None

    @classmethod
    def instantiate(cls, flask_app):
        if cls._instance is None:
            NarrativeGraphGenerator.instantiate()
            NodeContextCreator.instantiate()
            cls._instance = NarrativityServer()
            cls._instance.setup()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            raise Exception('Narrativity Server not instantiated.')
        return cls._instance

    def setup(self):
        self._narrative_graph_generator = NarrativeGraphGenerator.instance()
        self._node_context_creator = NodeContextCreator.instance()
        self._narrative_graph_generator.load()

    def upload_corpus(self, corpus):
        self._graph = self._narrative_graph_generator.generate(corpus["text"])

    def get_node_context(self, node_id):
        node = self._graph.id2node(node_id)
        node_context = self._node_context_creator.create(node)
        return node_context

    def get_most_connected_node(self):
        return list(self._graph.narrative_nodes().values())[0]
