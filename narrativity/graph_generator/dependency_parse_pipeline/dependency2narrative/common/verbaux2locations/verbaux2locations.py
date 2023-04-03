from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import resolve_compounds
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.creators import create_entity_node
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.extraction_paths.extraction_path_matcher import ExtractionPathMatcher
from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence
from narrativity.datamodel.narrative_graph.relationships.location_relationship import LocationRelationship


class VerbAux2Locations:
    def load(self):
        self._extraction_path_matcher = ExtractionPathMatcher()

    def convert(self, verb_token, all_children_tokens, narrative_node, narrative_graph):
        for child in all_children_tokens:
            path = FeaturizedSentence.dependency_path_between_tokens(verb_token, child)
            if self._extraction_path_matcher.match(path, 'location') is True:
                coreferences = child.coreference()
                preposition = self._get_location_preposition(child)
                if coreferences is not None:
                    for coreference in coreferences:
                        location_node = self._get_location_node(coreference, narrative_graph)
                else:
                    location_node = self._get_location_node(child, narrative_graph)
                self._add_location_relationship(location_node, narrative_node, preposition, narrative_graph)

    def _add_location_relationship(self, location_node, narrative_node, preposition, narrative_graph):
        location_relationship = LocationRelationship.create()
        location_relationship.set_narrative(narrative_node)
        location_relationship.set_narrative_graph(narrative_graph)
        location_relationship.set_location(location_node)
        location_relationship.set_preposition(preposition.text())
        narrative_node.add_location_relationship(location_relationship)
        location_node.add_narrative_relationship(location_relationship)
        narrative_graph.add_location_relationship(location_relationship)

    def _get_location_node(self, location_token, narrative_graph):
        whole_text = resolve_compounds(location_token)
        whole_text = ' '.join(i.text() for i in whole_text)
        location_node = narrative_graph.text2entity_node(whole_text)
        if location_node is not None:
            return location_node
        return create_entity_node(whole_text, narrative_graph)

    def _get_location_preposition(self, location_token):
        parent = location_token.parent()
        if parent.dep() == 'prep':
            return parent
        return None
