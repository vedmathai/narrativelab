from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative import Dependency2Narrative
from narrativity.graph_generator.dependency_parse_pipeline.corpus2spacy import Corpus2spacy 


class NarrativeGraphGenerator:
    def __init__(self):
        self._corpus2spacy = Corpus2spacy()
        self._dependency2narrative = Dependency2Narrative()

    def load(self):
        self._corpus2spacy.load()
        self._dependency2narrative.load()

    def generate(self, corpus):
        spacy_corpus = self._corpus2spacy.convert(corpus)
        self._dependency2narrative.convert(spacy_corpus)
