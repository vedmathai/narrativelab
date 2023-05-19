from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import resolve_compounds
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.creators import create_entity_node
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.extraction_paths.extraction_path_matcher import ExtractionPathMatcher
from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence
from narrativity.datamodel.narrative_graph.relationships.object_relationship import ObjectRelationship


class VerbAux2IndirectObjects:
    def load(self):
        self._extraction_path_matcher = ExtractionPathMatcher()

    def convert(self, verb_token, all_children_tokens, narrative_node, narrative_graph):
        for child in all_children_tokens:
            path = FeaturizedSentence.dependency_path_between_tokens(verb_token, child)
            if self._extraction_path_matcher.match(path, 'indirect_object') is True:
                coreferences = child.coreference()
                preposition = self._get_indirect_object_preposition(child)
                if coreferences is not None:
                    for coreference in coreferences:
                        indirect_object_node = self._get_indirect_object_node(coreference, narrative_graph)
                else:
                    indirect_object_node = self._get_indirect_object_node(child, narrative_graph)
                self._add_indirect_object_relationship(indirect_object_node, narrative_node, preposition, narrative_graph)

    def _add_indirect_object_relationship(self, indirect_object_node, narrative_node, preposition, narrative_graph):
        object_relationship = ObjectRelationship.create()
        object_relationship.set_narrative(narrative_node)
        object_relationship.set_narrative_graph(narrative_graph)
        object_relationship.set_object(indirect_object_node)
        object_relationship.set_preposition(preposition.text())
        narrative_node.add_indirect_object_relationship(object_relationship)
        indirect_object_node.add_narrative_relationship(object_relationship)
        narrative_graph.add_indirect_object_relationship(object_relationship)

    def _get_indirect_object_node(self, object_token, narrative_graph):
        whole_text = resolve_compounds(object_token)
        whole_text = ' '.join(i.text() for i in whole_text)
        object_node = narrative_graph.text2entity_node(whole_text)
        if object_node is not None:
            return object_node
        return create_entity_node(whole_text, narrative_graph)

    def _get_indirect_object_preposition(self, obj_token):
        parent = obj_token.parent()
        if parent.dep() == 'prep':
            return parent
        return None
