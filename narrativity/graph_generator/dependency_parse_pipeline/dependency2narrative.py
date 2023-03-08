import pprint

from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence
from narrativity.datamodel.narrative_graph.narrative_graph import NarrativeGraph
from narrativity.datamodel.narrative_graph.nodes.narrative_node import NarrativeNode
from narrativity.datamodel.narrative_graph.nodes.entity_node import EntityNode
from narrativity.datamodel.narrative_graph.nodes.action_node import ActionNode
from narrativity.datamodel.narrative_graph.relationships.object_relationship import ObjectRelationship


path2key = {
    ('nsubj', 'ROOT'): 'actor',
    ('advcl', 'nsubj'): 'actor',
    ('advcl', 'acomp'): 'state',
    ('acomp', 'prep'): 'state',
    ('acomp', 'prep', 'pobj'): 'state',
    ('acomp', 'prep', 'pobj', 'poss'): 'state',
}

verb_connectors = [
    ('ROOT', 'advcl'),
    ('ROOT', 'conj'),
]

verb2actor_paths = [
    ('ROOT', 'nsubj'),
    ('conj', 'nsubj'),
]

verb2direct_object_paths = [
    ('ROOT', 'dobj'),
    ('conj', 'dobj'),
]

class PhraseConnector:
    def __init__(self):
        self._verb_1 = None
        self._verb_2 = None
        self._connector_text = None
        self._connector_dep = None

    def verb_1(self):
        return self._verb_1

    def verb_2(self):
        return self._verb_2

    def connector_text(self):
        return self._connector_text

    def connector_dep(self):
        return self._connector_dep

    def set_verb_1(self, verb_1):
        self._verb_1 = verb_1

    def set_verb_2(self, verb_2):
        self._verb_2 = verb_2

    def set_connector_text(self, text):
        self._connector_text = text

    def set_connector_dep(self, dep):
        self._connector_dep = dep

    @staticmethod
    def create(verb_1, verb_2, connector_text, connector_dep):
        connector = PhraseConnector()
        connector.set_verb_1(verb_1)
        connector.set_verb_2(verb_2)
        connector.set_connector_text(connector_text)
        connector.set_connector_dep(connector_dep)
        return connector

class Dependency2Narrative:
    def __init__(self):
        self._narrative_graph = NarrativeGraph()

    def load(self):
        pass

    def convert(self, fdocument):
        phrase_connectors = []
        for sentence in fdocument.sentences():
            phrase_connectors = self.sentence2phrases(sentence.root(), phrase_connectors)
        for phrase_connector in phrase_connectors:
            verb_1 = phrase_connector.verb_1()
            verb_2 = phrase_connector.verb_2()
            self.clause_root2event(verb_1)
            self.clause_root2event(verb_2)
        pprint.pprint(self._narrative_graph.to_dict())

    def sentence2phrases(self, root, phrase_connectors):
        for dep, child_list in root.children().items():
            print(root.text(), [i.text() for i in child_list])
            for child in child_list:
                path = FeaturizedSentence.dependency_path_between_tokens(root, child)
                tup = (tuple(i.dep() for i in path))
                print(tup)
                if tup in verb_connectors:
                    phrase_connector = PhraseConnector.create(root, child, "", child.dep())
                    phrase_connectors.append(phrase_connector)
                    phrase_connectors = self.sentence2phrases(child, phrase_connectors)
        return phrase_connectors

    def clause_root2event(self, clause_root):
        print(clause_root.pos())
        if clause_root.pos() == 'VERB':
            self.verb2event(clause_root)
        if clause_root.pos() == 'AUX':
            self.aux2event(clause_root)

    def verb2event(self, verb):
        narrative_node = NarrativeNode.create()
        actor_nodes = self.verb2actors(verb, narrative_node)
        verb_nodes = self.verb2actions(verb, narrative_node)
        object_nodes = self.verb2direct_objects(verb, narrative_node)
        self._narrative_graph.add_narrative_node(narrative_node)
        
    
    def aux2event(self, aux):
        self._aux2actors(aux)
        self._aux2states(aux)

    def _aux2actors(self, aux):
        pass

    def _aux2states(self, aux):
        pass

    def verb2actions(self, verb, narrative_node):
        action_node = self.get_action_node(verb)
        narrative_node.add_action(action_node)

    def verb2actors(self, verb, narrative_node):
        for dep, child_list in verb.children().items():
            for child in child_list:
                path = FeaturizedSentence.dependency_path_between_tokens(verb, child)
                tup = (tuple(i.dep() for i in path))
                if tup in verb2actor_paths:
                    coreferences = child.coreference()
                    if coreferences is not None:
                        for coreference in coreferences:
                            actor_node = self.get_actor_node(coreference)
                    else:
                        actor_node = self.get_actor_node(child)
                    narrative_node.add_actor(actor_node)

    def verb2direct_objects(self, verb, narrative_node):
        for dep, child_list in verb.children().items():
            for child in child_list:
                path = FeaturizedSentence.dependency_path_between_tokens(verb, child)
                tup = (tuple(i.dep() for i in path))
                if tup in verb2direct_object_paths:
                    coreferences = child.coreference()
                    if coreferences is not None:
                        for coreference in coreferences:
                            direct_object_node = self.get_direct_object_node(coreference)
                    else:
                        direct_object_node = self.get_direct_object_node(child)
                    self.add_direct_object_relationship(direct_object_node, narrative_node)

    def add_direct_object_relationship(self, direct_object_node, narrative_node):
        object_relationship = ObjectRelationship.create()
        object_relationship.set_narrative(narrative_node)
        object_relationship.set_object(direct_object_node)
        narrative_node.add_direct_object_relationship(object_relationship)
        direct_object_node.add_narrative_relationship(object_relationship)
        self._narrative_graph.add_object_relationship(object_relationship)

    def get_action_node(self, verb_token):
        whole_text = self._resolve_auxiliaries(verb_token)
        whole_text = ' '.join(i.text() for i in whole_text)
        action_node = self._narrative_graph.text2action_node(whole_text)
        if action_node is not None:
            return action_node
        return self._create_action_node(whole_text)
                    
    def get_actor_node(self, actor_token):
        whole_text = self._resolve_compounds(actor_token)
        whole_text = ' '.join(i.text() for i in whole_text)
        actor_node = self._narrative_graph.text2entity_node(whole_text)
        if actor_node is not None:
            return actor_node
        return self._create_entity_node(whole_text)

    def get_direct_object_node(self, object_token):
        whole_text = self._resolve_compounds(object_token)
        whole_text = ' '.join(i.text() for i in whole_text)
        object_node = self._narrative_graph.text2entity_node(whole_text)
        if object_node is not None:
            return object_node
        return self._create_entity_node(whole_text)

    def _resolve_compounds(self, token):
        compound = []
        for dep, child_list in token.children().items():
            for child in child_list:
                if child.dep() == 'compound':
                    compound = self._resolve_compounds(child)
        compound.append(token)
        return compound

    def _resolve_auxiliaries(self, token):
        auxiliary = []
        for dep, child_list in token.children().items():
            for child in child_list:
                if child.dep() == 'auxiliary':
                    auxiliary = self._resolve_auxiliary(child)
        auxiliary.append(token)
        return auxiliary

    def _create_entity_node(self, whole_text):
        actor_node = EntityNode.create()
        actor_node.set_canonical_name(whole_text)
        self._narrative_graph.add_entity_node(actor_node)
        return actor_node

    def _create_action_node(self, whole_text):
        action_node = ActionNode.create()
        action_node.set_canonical_name(whole_text)
        self._narrative_graph.add_action_node(action_node)
        return action_node
