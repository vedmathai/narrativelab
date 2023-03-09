import pprint

from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.sentence2phrases import Sentence2Phrases
from narrativity.datamodel.narrative_graph.narrative_graph import NarrativeGraph
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.verb2event.verb2event import Verb2Event



path2key = {
    ('nsubj', 'ROOT'): 'actor',
    ('advcl', 'nsubj'): 'actor',
    ('advcl', 'acomp'): 'state',
    ('acomp', 'prep'): 'state',
    ('acomp', 'prep', 'pobj'): 'state',
    ('acomp', 'prep', 'pobj', 'poss'): 'state',
}

verb2actor_paths = [
    ('ROOT', 'nsubj'),
    ('conj', 'nsubj'),
]


class Dependency2Narrative:
    def __init__(self):
        self._sentence2phrases = Sentence2Phrases()
        self._verb2event = Verb2Event()

    def load(self):
        self._sentence2phrases.load()
        self._verb2event.load()

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
        if clause_root.pos() == 'VERB':
            self._verb2event.convert(clause_root, self._narrative_graph)
        if clause_root.pos() == 'AUX':
            self._aux2event(clause_root)
        
    
    def aux2event(self, aux):
        self._aux2actors(aux)
        self._aux2states(aux)

    def _aux2actors(self, aux):
        pass

    def _aux2states(self, aux):
        pass

