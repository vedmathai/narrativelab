from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import (
    resolve_compounds,
)
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.creators import (
    create_entity_node,
)
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.extraction_paths.extraction_path_matcher import ExtractionPathMatcher
from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence
from narrativity.datamodel.narrative_graph.relationships.subject_relationship import SubjectRelationship


class Aux2Subjects:

    def load(self):
        self._extraction_path_matcher = ExtractionPathMatcher()

    def convert(self, verb_token, all_children_tokens, narrative_node, narrative_graph):
        for child in all_children_tokens:
            path = FeaturizedSentence.dependency_path_between_tokens(verb_token, child)
            if self._extraction_path_matcher.match(path, 'subject') is True:
                coreferences = child.coreference()
                if coreferences is not None:
                    for coreference in coreferences:
                        subject_node = self.get_subject_node(coreference, narrative_graph)
                else:
                    subject_node = self.get_subject_node(child, narrative_graph)
                self.add_subject_relationship(subject_node, narrative_node, narrative_graph)

    def add_subject_relationship(self, subject_node, narrative_node, narrative_graph):
        subject_relationship = SubjectRelationship.create()
        subject_relationship.set_narrative_graph(narrative_graph)
        subject_relationship.set_narrative(narrative_node)
        subject_relationship.set_subject(subject_node)
        narrative_node.add_subject_relationship(subject_relationship)
        subject_node.add_narrative_relationship(subject_relationship)
        narrative_graph.add_subject_relationship(subject_relationship)

    def get_subject_node(self, subject_token, narrative_graph):
        whole_text = resolve_compounds(subject_token)
        whole_text = ' '.join(i.text() for i in whole_text)
        subject_node = narrative_graph.text2entity_node(whole_text)
        if subject_node is not None:
            return subject_node
        return create_entity_node(whole_text, narrative_graph)
