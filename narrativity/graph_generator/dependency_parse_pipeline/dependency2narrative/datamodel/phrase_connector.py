class PhraseConnector:
    def __init__(self):
        self._verb_1 = None
        self._verb_2 = None
        self._connector_type = None
        self._connector_text = None
        self._is_main = None

    def verb_1(self):
        return self._verb_1

    def verb_2(self):
        return self._verb_2

    def connector_type(self):
        return self._connector_type

    def connector_text(self):
        return self._connector_text
    
    def is_main(self):
        return self._is_main

    def set_verb_1(self, verb_1):
        self._verb_1 = verb_1

    def set_verb_2(self, verb_2):
        self._verb_2 = verb_2

    def set_connector_type(self, connector_type):
        self._connector_type = connector_type

    def set_connector_text(self, connector_text):
        self._connector_text = connector_text

    def set_is_main(self, is_main):
        self._is_main = is_main

    @staticmethod
    def create(verb_1, verb_2, connector_type, connector_text, is_main):
        connector = PhraseConnector()
        connector.set_verb_1(verb_1)
        connector.set_verb_2(verb_2)
        connector.set_connector_type(connector_type)
        connector.set_connector_text(connector_text)
        connector.set_is_main(is_main)
        return connector

    def to_dict(self):
        return {
            "verb_1": self.verb_1().text(),
            "verb_2": self.verb_2().text(),
            "connector_type": self.connector_type(),
            "connector_text": self.connector_text().text(),
            "is_main": self.is_main(),
        }