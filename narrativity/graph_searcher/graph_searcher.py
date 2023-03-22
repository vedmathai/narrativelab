from narrativity.graph_searcher.datamodel.search_response.search_response import SearchResponse


class GraphSearcher:

    _instance = None
    _name = "Graph Searcher"

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

    def search(self, search_request, narrative_graph):
        search_response = SearchResponse()
        narrative_nodes = list(narrative_graph.narrative_nodes().values())
        search_response.set_nodes(narrative_nodes)
        return search_response
