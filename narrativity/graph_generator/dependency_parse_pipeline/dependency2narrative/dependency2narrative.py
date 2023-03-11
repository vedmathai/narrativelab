import pprint

from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.sentence2phrases import Sentence2Phrases
from narrativity.datamodel.narrative_graph.narrative_graph import NarrativeGraph
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.verb2event.verb2event import Verb2Event
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.aux2state.aux2state import Aux2State


class Dependency2Narrative:
    def __init__(self):
        self._sentence2phrases = Sentence2Phrases()
        self._verb2event = Verb2Event()
        self._aux2state = Aux2State()

    def load(self):
        self._sentence2phrases.load()
        self._verb2event.load()
        self._aux2state.load()

    def convert(self, fdocument):
        self._narrative_graph = NarrativeGraph()
        phrase_connectors = []
        for sentence in fdocument.sentences():
            phrase_connectors = self._sentence2phrases.split(sentence.root(), phrase_connectors)
        for phrase_connector in phrase_connectors:
            verb_1 = phrase_connector.verb_1()
            verb_2 = phrase_connector.verb_2()
            self.clause_root2event(verb_1)
            self.clause_root2event(verb_2)
        return self._narrative_graph

    def clause_root2event(self, clause_root):
        if clause_root is None:
            return
        if clause_root.pos() == 'VERB':
            self._verb2event.convert(clause_root, self._narrative_graph)
        print(clause_root.pos())
        if clause_root.pos() == 'AUX':
            self._aux2state.convert(clause_root, self._narrative_graph)
    