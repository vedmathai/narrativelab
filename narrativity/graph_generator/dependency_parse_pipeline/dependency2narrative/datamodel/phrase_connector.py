class PhraseConnector:
    def __init__(self):
        self._verb_1 = None
        self._verb_2 = None
        self._connector_text = None
        self._connector_dep = None

    def verb_1(self):
        return self._verb_1

    def verb_2(self):
        return self._verb_2

    def connector_text(self):
        return self._connector_text

    def connector_dep(self):
        return self._connector_dep

    def set_verb_1(self, verb_1):
        self._verb_1 = verb_1

    def set_verb_2(self, verb_2):
        self._verb_2 = verb_2

    def set_connector_text(self, text):
        self._connector_text = text

    def set_connector_dep(self, dep):
        self._connector_dep = dep

    @staticmethod
    def create(verb_1, verb_2, connector_text, connector_dep):
        connector = PhraseConnector()
        connector.set_verb_1(verb_1)
        connector.set_verb_2(verb_2)
        connector.set_connector_text(connector_text)
        connector.set_connector_dep(connector_dep)
        return connector