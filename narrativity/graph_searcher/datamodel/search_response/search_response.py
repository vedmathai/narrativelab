class SearchResponse:
    def __init__(self):
        self._narrative_nodes = []
        self._entity_nodes = []
        self._absolute_temporal_nodes = []

    def narrative_nodes(self):
        return self._narrative_nodes

    def add_narrative_node(self, narrative_node):
        self._narrative_nodes.append(narrative_node)

    def set_narrative_nodes(self, narrative_nodes):
        self._narrative_nodes = narrative_nodes

    def entity_nodes(self):
        return self._entity_nodes

    def add_entity_node(self, entity_node):
        self._entity_nodes.append(entity_node)

    def set_entity_nodes(self, entity_nodes):
        self._entity_nodes = entity_nodes

    def absolute_temporal_nodes(self):
        return self._absolute_temporal_nodes

    def add_absolute_temporal_node(self, absolute_temporal_node):
        self._absolute_temporal_nodes.append(absolute_temporal_node)

    def set_absolute_temporal_nodes(self, absolute_temporal_nodes):
        self._absolute_temporal_nodes = absolute_temporal_nodes

    def to_dict(self):
        return {
            'narrative_nodes': [i.to_dict() for i in self.narrative_nodes()],
            'entity_nodes': [i.to_dict() for i in self.entity_nodes()],
            'absolute_temporal_nodes': [i.to_dict() for i in self.absolute_temporal_nodes()],
        }
