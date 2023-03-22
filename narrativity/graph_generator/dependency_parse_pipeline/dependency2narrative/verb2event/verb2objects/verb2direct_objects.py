from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import resolve_compounds
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.creators import create_entity_node
from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence
from narrativity.datamodel.narrative_graph.relationships.object_relationship import ObjectRelationship


verb2direct_object_paths = [
    ('ROOT', 'dobj'),
    ('conj', 'dobj'),
    ('ROOT', 'nsubjpass'),
]

class Verb2DirectObjects:
    def load(self):
        pass

    def convert(self, verb_token, all_children_tokens, narrative_node, narrative_graph):
        for child in all_children_tokens:
            path = FeaturizedSentence.dependency_path_between_tokens(verb_token, child)
            tup = (tuple(i.dep() for i in path))
            if tup in verb2direct_object_paths:
                coreferences = child.coreference()
                if coreferences is not None:
                    for coreference in coreferences:
                        direct_object_node = self.get_direct_object_node(coreference, narrative_graph)
                else:
                    direct_object_node = self.get_direct_object_node(child, narrative_graph)
                self.add_direct_object_relationship(direct_object_node, narrative_node, narrative_graph)

    def add_direct_object_relationship(self, direct_object_node, narrative_node, narrative_graph):
        object_relationship = ObjectRelationship.create()
        object_relationship.set_narrative(narrative_node)
        object_relationship.set_object(direct_object_node)
        object_relationship.set_narrative_graph(narrative_graph)
        narrative_node.add_direct_object_relationship(object_relationship)
        direct_object_node.add_narrative_relationship(object_relationship)
        narrative_graph.add_direct_object_relationship(object_relationship)

    def get_direct_object_node(self, object_token, narrative_graph):
        whole_text = resolve_compounds(object_token)
        whole_text = ' '.join(i.text() for i in whole_text)
        object_node = narrative_graph.text2entity_node(whole_text)
        print(type(object_node))
        if object_node is not None:
            return object_node
        return create_entity_node(whole_text, narrative_graph)
