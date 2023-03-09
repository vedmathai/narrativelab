from narrativity.datamodel.narrative_graph.nodes.entity_node import EntityNode
from narrativity.datamodel.narrative_graph.nodes.action_node import ActionNode


def create_entity_node(whole_text, narrative_graph):
    actor_node = EntityNode.create()
    actor_node.set_canonical_name(whole_text)
    narrative_graph.add_entity_node(actor_node)
    return actor_node

def create_action_node(whole_text, narrative_graph):
    action_node = ActionNode.create()
    action_node.set_canonical_name(whole_text)
    narrative_graph.add_action_node(action_node)
    return action_node
