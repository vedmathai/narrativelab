class NodeContext:
    def __init__(self):
        self._node = None
        self._id2node = {}
        self._id2relationship = {}
        self._keys = []
        self._key2ids = {}

    def node(self):
        return self._node

    def set_node(self, node):
        self._node = node

    def id2node(self):
        return self._id2node

    def id2relationship(self):
        return self._id2relationship

    def add_key2id(self, key, id):
        if key not in self._key2ids:
            self._key2ids[key] = []
        self._key2ids[key].append(id)

    def add_key(self, key):
        self._keys.append(key)

    def keys(self):
        # Important to maintain ordering of the row order displayed
        return self._keys

    def key2ids(self):
        return self._key2ids

    def add_id2node(self, id, node):
        self._id2node[id] = node

    def add_id2relationship(self, id, relationship):
        self._id2relationship[id] = relationship

    def to_dict(self):
        return {
            "node": self.node().to_dict(),
            "id2node": {k: v.to_dict() for k, v in self.id2node().items()},
            "id2relationship": {k: v.to_dict() for k, v in self.id2relationship().items()},
            "keys": self.keys(),
            "key2id": self.key2ids()
        }