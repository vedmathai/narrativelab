from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import resolve_absolute_time_compounds
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.creators import create_absolute_temporal_node
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.extraction_paths.extraction_path_matcher import ExtractionPathMatcher
from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence
from narrativity.datamodel.narrative_graph.relationships.absolute_temporal_relationship import AbsoluteTemporalRelationship


class VerbAux2AbsoluteTimes:
    def load(self):
        self._extraction_path_matcher = ExtractionPathMatcher()

    def convert(self, verb_token, all_children_tokens, narrative_node, narrative_graph):
        for child in all_children_tokens:
            path = FeaturizedSentence.dependency_path_between_tokens(verb_token, child)
            if self._extraction_path_matcher.match(path, 'absolute_times') is True:
                absolute_temporal_node = self.get_absolute_temporal_node(child, narrative_graph)
                preposition = self._get_absolute_temporal_preposition(child)
                self.add_absolute_time_relationship(absolute_temporal_node, narrative_node, preposition, narrative_graph)

    def add_absolute_time_relationship(self, absolute_temporal_node, narrative_node, preposition, narrative_graph):
        absolute_temporal_relationship = AbsoluteTemporalRelationship.create()
        absolute_temporal_relationship.set_narrative(narrative_node)
        absolute_temporal_relationship.set_absolute_temporal_node(absolute_temporal_node)
        absolute_temporal_relationship.set_narrative_graph(narrative_graph)
        absolute_temporal_relationship.set_preposition(preposition.text())
        narrative_node.add_absolute_temporal_relationship(absolute_temporal_relationship)
        absolute_temporal_node.add_narrative_relationship(absolute_temporal_relationship)
        narrative_graph.add_absolute_temporal_relationship(absolute_temporal_relationship)

    def get_absolute_temporal_node(self, time_token, narrative_graph):
        whole_text = resolve_absolute_time_compounds(time_token)
        whole_text = ' '.join(i.text() for i in whole_text)
        temporal_node = narrative_graph.text2absolute_temporal_node(whole_text)
        if temporal_node is not None:
            return temporal_node
        return create_absolute_temporal_node(whole_text, narrative_graph)

    def _get_absolute_temporal_preposition(self, obj_token):
        parent = obj_token.parent()
        if parent.dep() == 'prep':
            return parent
        return None