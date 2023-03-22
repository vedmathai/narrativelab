from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.datamodel.phrase_connector import PhraseConnector
from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import get_all_children_tokens
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.extraction_paths.extraction_path_matcher import ExtractionPathMatcher


class Sentence2Phrases:

    def load(self):
        self._extraction_path_matcher = ExtractionPathMatcher()

    def split(self, root, phrase_connectors, is_root=True):
        single_root_flag = True
        all_children_tokens = get_all_children_tokens(root)
        for child in all_children_tokens:
            path = FeaturizedSentence.dependency_path_between_tokens(root, child)
            if self._extraction_path_matcher.match(path, 'causation') is True:
                single_root_flag = False
                mark = child
                parent_verb = self._find_parent_clause(mark)
                phrase_connector = PhraseConnector.create(root, parent_verb, "causation", mark)
                phrase_connectors.append(phrase_connector)
                phrase_connectors = self.split(child, phrase_connectors, False)
            if self._extraction_path_matcher.match(path, 'contradiction_detection') is True:
                single_root_flag = False
                mark = child
                for child2 in all_children_tokens:
                    path2 = FeaturizedSentence.dependency_path_between_tokens(mark, child2)
                    if self._extraction_path_matcher.match(path2, 'contradiction') is True:
                        second_verb = child2
                        phrase_connector = PhraseConnector.create(root, second_verb, "contradiction", mark)
                        phrase_connectors.append(phrase_connector)
                        phrase_connectors = self.split(child2, phrase_connectors, False)
        if is_root is True and single_root_flag is True:
            phrase_connector = PhraseConnector.create(root, None, "single", None)
            phrase_connectors.append(phrase_connector)
        return phrase_connectors

    def _find_parent_clause(self, mark):
        current = mark
        while current.pos() not in ['AUX', 'VERB']:
            current = current.parent()
        return current
