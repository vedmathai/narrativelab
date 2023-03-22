from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import (
    resolve_compounds,
)
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.creators import (
    create_entity_node,
)
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.extraction_paths.extraction_path_matcher import ExtractionPathMatcher
from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence
from narrativity.datamodel.narrative_graph.relationships.actor_relationship import ActorRelationship


class Verb2Actors:

    def load(self):
        self._extraction_path_matcher = ExtractionPathMatcher()

    def convert(self, verb_token, all_children_tokens, narrative_node, narrative_graph):
        for child in all_children_tokens:
            path = FeaturizedSentence.dependency_path_between_tokens(verb_token, child)
            if self._extraction_path_matcher.match(path, 'actor') is True:
                coreferences = child.coreference()
                if coreferences is not None:
                    for coreference in coreferences:
                        actor_node = self.get_actor_node(coreference, narrative_graph)
                else:
                    actor_node = self.get_actor_node(child, narrative_graph)
                self.add_actor_relationship(actor_node, narrative_node, narrative_graph)

    def add_actor_relationship(self, actor_node, narrative_node, narrative_graph):
        actor_relationship = ActorRelationship.create()
        actor_relationship.set_narrative_graph(narrative_graph)
        actor_relationship.set_narrative(narrative_node)
        actor_relationship.set_actor(actor_node)
        narrative_node.add_actor_relationship(actor_relationship)
        actor_node.add_narrative_relationship(actor_relationship)
        narrative_graph.add_actor_relationship(actor_relationship)

    def get_actor_node(self, actor_token, narrative_graph):
        whole_text = resolve_compounds(actor_token)
        whole_text = ' '.join(i.text() for i in whole_text)
        actor_node = narrative_graph.text2entity_node(whole_text)
        if actor_node is not None:
            return actor_node
        return create_entity_node(whole_text, narrative_graph)
