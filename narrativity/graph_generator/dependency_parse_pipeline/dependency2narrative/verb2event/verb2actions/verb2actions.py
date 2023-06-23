from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.utils import (
    resolve_auxiliaries, get_main_verb
)
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.common.creators import (
    create_action_node,
)


class Verb2Actions:
    def load(self):
        pass

    def convert(self, verb_token, all_children_tokens, narrative_node, narrative_graph):
        action_node = self.get_action_node(verb_token, narrative_graph)
        narrative_node.add_action(action_node)
        self.check_negative(verb_token, narrative_node)

    def get_action_node(self, verb_token, narrative_graph):
        whole_text = resolve_auxiliaries(verb_token)
        whole_text = ' '.join(i.text() for i in whole_text)
        action_node = narrative_graph.text2action_node(whole_text)
        if action_node is not None:
            return action_node
        action_node = create_action_node(whole_text, narrative_graph)
        return action_node

    def check_negative(self, verb_token, narrative_node):
        if 'neg' in verb_token.children():
            narrative_node.set_is_negative(True)
