class ExtractionPathElement:
    def __init__(self):
        self._deps = []
        self._pos = []
        self._tokens = []
        self._tokens_blacklist = []

    def deps(self):
        return self._deps

    def pos(self):
        return self._pos

    def tokens(self):
        return self._tokens

    def tokens_blacklist(self):
        return self._tokens_blacklist

    def set_deps(self, deps):
        self._deps = deps

    def set_pos(self, pos):
        self._pos = pos

    def set_tokens(self, tokens):
        self._tokens = tokens

    def set_tokens_blacklist(self, tokens_blacklist):
        self._tokens_blacklist = tokens_blacklist

    def to_dict(self):
        return {
            "deps": self.deps(),
            "pos": self.pos(),
            "tokens": self.tokens(),
            "tokens_blacklist": self.tokens_blacklist(),
        }

    @staticmethod
    def from_dict(val):
        extraction_path_element = ExtractionPathElement()
        extraction_path_element.set_deps(val['deps'])
        extraction_path_element.set_pos(val['pos'])
        extraction_path_element.set_tokens(val['tokens'])
        extraction_path_element.set_tokens_blacklist(val['tokens_blacklist'])
        return extraction_path_element
