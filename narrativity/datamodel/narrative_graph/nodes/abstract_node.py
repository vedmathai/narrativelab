class AbstractNode:
    _abstract_type = "node"

    def __init__(self):
        self._id = None
        self._narrative_graph = None

    def type(self):
        return self._type
    
    def abstract_type(self):
        return self._abstract_type

    def id(self):
        return self._id

    def set_id(self, id: str):
        self._id = id

    def set_narrative_graph(self, narrative_graph) -> None:
        self._narrative_graph = narrative_graph

    def relationships(self):
        return sum(self._relationships, [])
