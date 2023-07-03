class AbstractRelationship:
    _abstract_type = "relationship"

    def __init__(self):
        self._id = None
        self._narrative_graph = None
    
    def abstract_type(self):
        return self._abstract_type

    def type(self):
        return self._type

    def id(self):
        return self._id

    def set_id(self, id: str):
        self._id = id

    def set_narrative_graph(self, narrative_graph) -> None:
        self._narrative_graph = narrative_graph

    def nodes(self):
        return self._nodes
    
    def display_name(self):
        return None
