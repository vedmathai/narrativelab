from narrativity.datamodel.narrative_graph.nodes.entity_node import EntityNode
from narrativity.datamodel.narrative_graph.nodes.action_node import ActionNode


def create_entity_node(whole_text, narrative_graph):
    entity_node = EntityNode.create()
    entity_node.set_canonical_name(whole_text)
    narrative_graph.add_entity_node(entity_node)
    entity_node.set_narrative_graph(narrative_graph)
    return entity_node

def create_action_node(whole_text, narrative_graph):
    action_node = ActionNode.create()
    action_node.set_canonical_name(whole_text)
    narrative_graph.add_action_node(action_node)
    return action_node
