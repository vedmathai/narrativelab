from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.datamodel.phrase_connector import PhraseConnector
from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import get_all_children_tokens
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.extraction_paths.extraction_path_matcher import ExtractionPathMatcher


class Sentence2Phrases:

    def load(self):
        self._extraction_path_matcher = ExtractionPathMatcher()

    def split(self, root, phrase_connectors, is_main=False, is_root=True):
        single_root_flag = True
        all_children_tokens = get_all_children_tokens(root)
        for child in all_children_tokens:
            path = FeaturizedSentence.dependency_path_between_tokens(root, child)
            phrase_connector_fns = [
                self._anecdotal_relationship,
                self._backwards_contradiction,
                self._forwards_contradiction,
                self._but_like_contradiction,
                self._however_like_contradiction,
                self._after_like_temporal_relationship,
                self._and_like_relationship,
                self._prep_relationship,
                self._descriptor_relationship,
            ]
            for phrase_connector_fn in phrase_connector_fns:
                phrase_connectors, single_root_flag = phrase_connector_fn(path, phrase_connectors, root, child, all_children_tokens, single_root_flag, is_main)

        if self._check_single_clause(is_root, single_root_flag, root) is True:
            phrase_connector = PhraseConnector.create(root, None, "single", None, is_main)
            phrase_connectors.append(phrase_connector)
        return phrase_connectors

    def _check_single_clause(self, is_root, single_root_flag, root):
        verb_like_check = root.pos() in ['VERB', 'AUX']
        return all([is_root, single_root_flag, verb_like_check])

    def _find_parent_clause(self, mark):
        current = mark
        while current.pos() not in ['AUX', 'VERB']:
            current = current.parent()
        return current

    def _anecdotal_relationship(self, path, phrase_connectors, root, child, all_children_tokens, single_root_flag, is_main):
        if self._extraction_path_matcher.match(path, 'anecdotal_relationship'):
            single_root_flag = False
            phrase_connector = PhraseConnector.create(root, child, "anecdotal_relationship", None, is_main)
            phrase_connectors.append(phrase_connector)
            phrase_connectors = self.split(child, phrase_connectors, False)
        return phrase_connectors, single_root_flag

    def _forwards_contradiction(self, path, phrase_connectors, root, child, all_children_tokens, single_root_flag, is_main):
        if self._extraction_path_matcher.match(path, 'forwards_causation') is True:
            single_root_flag = False
            mark = child
            parent_verb = self._find_parent_clause(mark)
            phrase_connector = PhraseConnector.create(root, parent_verb, "forwards_causation", mark, is_main)
            phrase_connectors.append(phrase_connector)
            phrase_connectors = self.split(child, phrase_connectors, False)
        return phrase_connectors, single_root_flag

    def _backwards_contradiction(self, path, phrase_connectors, root, child, all_children_tokens, single_root_flag, is_main):
        if self._extraction_path_matcher.match(path, 'backwards_causation') is True:
            single_root_flag = False
            mark = child
            parent_verb = self._find_parent_clause(mark)
            phrase_connector = PhraseConnector.create(root, parent_verb, "backwards_causation", mark, is_main)
            phrase_connectors.append(phrase_connector)
            phrase_connectors = self.split(child, phrase_connectors, False)
        return phrase_connectors, single_root_flag

    def _but_like_contradiction(self, path, phrase_connectors, root, child, all_children_tokens, single_root_flag, is_main):
        if self._extraction_path_matcher.match(path, 'but_like_contradiction_detection') is True:
            single_root_flag = False
            mark = child
            for child2 in all_children_tokens:
                path2 = FeaturizedSentence.dependency_path_between_tokens(mark, child2)
                if self._extraction_path_matcher.match(path2, 'but_like_contradiction') is True:
                    second_verb = child2
                    phrase_connector = PhraseConnector.create(root, second_verb, "but_like_contradiction", mark, is_main)
                    phrase_connectors.append(phrase_connector)
                    phrase_connectors = self.split(child2, phrase_connectors, False)
        return phrase_connectors, single_root_flag
        
    def _however_like_contradiction(self, path, phrase_connectors, root, child, all_children_tokens, single_root_flag, is_main):
        if self._extraction_path_matcher.match(path, 'however_like_contradiction_detection') is True:
            single_root_flag = False
            mark = child
            for child2 in all_children_tokens:
                path2 = FeaturizedSentence.dependency_path_between_tokens(mark, child2)
                if self._extraction_path_matcher.match(path2, 'however_like_contradiction') is True:
                    second_verb = child2
                    phrase_connector = PhraseConnector.create(root, second_verb, "however_like_contradiction", mark, is_main)
                    phrase_connectors.append(phrase_connector)
                    phrase_connectors = self.split(child2, phrase_connectors, False)
        return phrase_connectors, single_root_flag

    def _after_like_temporal_relationship(self, path, phrase_connectors, root, child, all_children_tokens, single_root_flag, is_main):
        if self._extraction_path_matcher.match(path, 'after_like_temporal_relationship_detection') is True:
            single_root_flag = False
            mark = child
            for child2 in all_children_tokens:
                path2 = FeaturizedSentence.dependency_path_between_tokens(mark, child2)
                if self._extraction_path_matcher.match(path2, 'after_like_temporal_relationship') is True:
                    second_verb = child2
                    phrase_connector = PhraseConnector.create(root, second_verb, "after_like_temporal_relationship", mark, is_main)
                    phrase_connectors.append(phrase_connector)
                    phrase_connectors = self.split(child2, phrase_connectors, False)
        return phrase_connectors, single_root_flag

    def _and_like_relationship(self, path, phrase_connectors, root, child, all_children_tokens, single_root_flag, is_main):
        if self._extraction_path_matcher.match(path, 'and_like_relationship') is True:
            single_root_flag = False
            phrase_connector = PhraseConnector.create(root, child, "and_like_relationship", None, is_main)
            phrase_connectors.append(phrase_connector)
            phrase_connectors = self.split(child, phrase_connectors, False)
        return phrase_connectors, single_root_flag

    def _prep_relationship(self, path, phrase_connectors, root, child, all_children_tokens, single_root_flag, is_main):
        if self._extraction_path_matcher.match(path, 'prep_relationship') is True:
            single_root_flag = False
            prep = child.parent()
            phrase_connector = PhraseConnector.create(root, child, "prep_relationship", prep, is_main)
            phrase_connectors.append(phrase_connector)
            phrase_connectors = self.split(child, phrase_connectors, False)
        return phrase_connectors, single_root_flag

    def _descriptor_relationship(self, path, phrase_connectors, root, child, all_children_tokens, single_root_flag, is_main):
        if self._extraction_path_matcher.match(path, 'descriptor_relationship_detection') is True:
            single_root_flag = False
            object = child.parent()
            phrase_connector = PhraseConnector.create(root, child, "descriptor_relationship", object, is_main)
            phrase_connectors.append(phrase_connector)
            phrase_connectors = self.split(child, phrase_connectors, False)
        return phrase_connectors, single_root_flag
