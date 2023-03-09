from narrativity.graph_generator.dependency_parse_pipeline.parser import NarrativeGraphGenerator


class NarrativityServer:

    def __init__(self):
        self._corpus = None

    _instance = None

    @classmethod
    def instantiate(cls, flask_app):
        if cls._instance is None:
            NarrativeGraphGenerator.instantiate()
            cls._instance = NarrativityServer()
            cls._instance.setup()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            raise Exception('Narrativity Server not instantiated.')
        return cls._instance

    def setup(self):
        self._narrative_graph_generator = NarrativeGraphGenerator.instance()
        self._narrative_graph_generator.load()

    def upload_corpus(self, corpus):
        self._graph = self._narrative_graph_generator.generate(corpus["text"])

    def get_nodes(self):
        nodes = [i.canonical_name() for i in self._graph.entity_nodes().values()]
        print(nodes)
        return nodes