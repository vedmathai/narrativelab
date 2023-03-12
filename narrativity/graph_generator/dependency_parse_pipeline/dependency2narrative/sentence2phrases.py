from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.datamodel.phrase_connector import PhraseConnector
from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import get_all_children_tokens


verb_connectors = [
    ('ROOT', 'advcl'),
    ('ROOT', 'conj'),
    ('ROOT', 'acomp', 'prep', 'pobj')
]

class Sentence2Phrases:

    def load(self):
        pass

    def split(self, root, phrase_connectors, is_root=True):
        single_root_flag = True
        all_children_tokens = get_all_children_tokens(root)
        for child in all_children_tokens:
            path = FeaturizedSentence.dependency_path_between_tokens(root, child)
            tup = (tuple(i.dep() for i in path))
            if tup in verb_connectors:
                single_root_flag = False
                phrase_connector = PhraseConnector.create(root, child, "", child.dep())
                phrase_connectors.append(phrase_connector)
                phrase_connectors = self.split(child, phrase_connectors, False)
        if is_root is True and single_root_flag is True:
            phrase_connector = PhraseConnector.create(root, None, "", None)
            phrase_connectors.append(phrase_connector)
        return phrase_connectors