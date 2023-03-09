from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import (
    resolve_compounds,
)
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.creators import (
    create_entity_node,
)
from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence


verb2actor_paths = [
    ('ROOT', 'nsubj'),
    ('conj', 'nsubj'),
]

class Verb2Actors:

    def load(self):
        pass

    def convert(self, verb_token, all_children_tokens, narrative_node, narrative_graph):
        for child in all_children_tokens:
            path = FeaturizedSentence.dependency_path_between_tokens(verb_token, child)
            tup = (tuple(i.dep() for i in path))
            if tup in verb2actor_paths:
                coreferences = child.coreference()
                if coreferences is not None:
                    for coreference in coreferences:
                        actor_node = self.get_actor_node(coreference, narrative_graph)
                else:
                    actor_node = self.get_actor_node(child, narrative_graph)
                narrative_node.add_actor(actor_node)

    def get_actor_node(self, actor_token, narrative_graph):
        whole_text = resolve_compounds(actor_token)
        whole_text = ' '.join(i.text() for i in whole_text)
        actor_node = narrative_graph.text2entity_node(whole_text)
        if actor_node is not None:
            return actor_node
        return create_entity_node(whole_text, narrative_graph)
