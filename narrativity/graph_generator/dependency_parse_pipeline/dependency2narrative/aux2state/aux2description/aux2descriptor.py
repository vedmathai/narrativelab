from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import resolve_compounds
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.creators import create_entity_node
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.extraction_paths.extraction_path_matcher import ExtractionPathMatcher
from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence
from narrativity.datamodel.narrative_graph.relationships.state_relationship import StateRelationship


class Aux2StateDescriptor:
    def load(self):
        self._extraction_path_matcher = ExtractionPathMatcher()

    def convert(self, aux_token, all_children_tokens, narrative_node, narrative_graph):
        for child in all_children_tokens:
            path = FeaturizedSentence.dependency_path_between_tokens(aux_token, child)
            if self._extraction_path_matcher.match(path, 'state_desciptor') is True:
                coreferences = child.coreference()
                if coreferences is not None:
                    for coreference in coreferences:
                        direct_object_node = self.get_state_node(coreference, narrative_graph)
                else:
                    direct_object_node = self.get_state_node(child, narrative_graph)
                self.add_state_relationship(direct_object_node, aux_token, narrative_node, narrative_graph)

    def add_state_relationship(self, state_node, aux, narrative_node, narrative_graph):
        state_relationship = StateRelationship.create()
        state_relationship.set_narrative(narrative_node)
        state_relationship.set_narrative_graph(narrative_graph)
        state_relationship.set_state(state_node)
        state_relationship.set_auxiliary(aux.text())
        narrative_node.add_state_relationship(state_relationship)
        state_node.add_narrative_relationship(state_relationship)
        narrative_graph.add_state_relationship(state_relationship)

    def get_state_node(self, state_token, narrative_graph):
        whole_text_tokens = resolve_compounds(state_token)
        whole_text = ' '.join(i.text() for i in whole_text_tokens)
        state_node = narrative_graph.text2entity_node(whole_text)
        if state_node is not None:
            return state_node
        return create_entity_node(whole_text, narrative_graph)
