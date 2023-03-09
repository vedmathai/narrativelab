from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.datamodel.phrase_connector import PhraseConnector
from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence

verb_connectors = [
    ('ROOT', 'advcl'),
    ('ROOT', 'conj'),
]

class Sentence2Phrases:

    def load(self):
        pass

    def split(self, root, phrase_connectors):
        for dep, child_list in root.children().items():
            for child in child_list:
                path = FeaturizedSentence.dependency_path_between_tokens(root, child)
                tup = (tuple(i.dep() for i in path))
                if tup in verb_connectors:
                    phrase_connector = PhraseConnector.create(root, child, "", child.dep())
                    phrase_connectors.append(phrase_connector)
                    phrase_connectors = self.split(child, phrase_connectors)
        return phrase_connectors