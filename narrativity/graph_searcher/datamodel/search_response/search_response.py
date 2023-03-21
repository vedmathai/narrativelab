class SearchResponse:
    def __init__(self):
        self._nodes = []

    def nodes(self):
        return self._nodes

    def add_node(self, node):
        self._nodes.append(node)

    def set_nodes(self, nodes):
        self._nodes = nodes

    def to_dict(self):
        return {
            'nodes': [i.to_dict() for i in self.nodes()],
        }
