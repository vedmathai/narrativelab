from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import uuid
from collections import defaultdict

class Node:
    def __init__(self):
        self._id = str(uuid.uuid4())
        self._type = 'entity'
        self._token = None
        self._text = None
        self._relationships = []

    def __str__(self):
        return "Node(text={})".format(self._text)

class Relationship:
    def __init__(self):
        self._id = str(uuid.uuid4())
        self._text = ''
        self._type = None
        self._node_1 = None
        self._node_2 = None

    def __str__(self):
        return "Relationship(node_1={} type={} node_2={})".format(self._node_1, self._type, self._node_2)

predictor = Predictor.from_path("/home/lalady6977/oerc/projects/local_jade/jade_front/narrative-lab/data/structured-prediction-srl-bert.2020.12.15.tar.gz")
p = predictor.predict(
    sentence="Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 ."
)
print(p)
string2node = {}

srl_dict = p
verb_locs = []
words = srl_dict['words']
relationships = []
previous_tag = ""
verb_nodes = []
for verb in srl_dict['verbs']:
    current_index = []
    tag2nodes = defaultdict(list)
    for tag_i, tag in  enumerate(verb['tags']):
        print(tag)
        if tag[0] == 'B':
            if len(current_index) > 0:
                string = ' '.join(words[i] for i in current_index)
                if string not in string2node:
                    node = Node()
                    node._text = string
                    string2node[string] = node
                node = string2node[string]
                tag2nodes[previous_tag].append(node)
            current_index = [tag_i]
            previous_tag = tag[2:]
        elif tag[0] == 'O':
            if len(current_index) > 0:
                string = ' '.join(words[i] for i in current_index)
                if string not in string2node:
                    node = Node()
                    node._text = string
                    string2node[string] = node
                node = string2node[string]
                tag2nodes[previous_tag].append(node)
                current_index = []
                previous_tag = ""
        else:
            current_index.append(tag_i)
    verb_node = tag2nodes['V']
    verb_nodes.extend(verb_node)
    if len(verb_node) > 0:
        for tag in tag2nodes:
            if tag != 'V':
                nodes = tag2nodes[tag]
                for node in nodes:
                    relationship = Relationship()
                    relationship._node_1 = verb_node[0]
                    relationship._node_2 = node
                    relationship._type = tag
                    relationships.append(relationship)

for verb_node_1_i, verb_node_1 in enumerate(verb_nodes):
    for verb_node_2_i, verb_node_2 in enumerate(verb_nodes):
        if verb_node_1_i < verb_node_2_i:
            relationship = Relationship()
            relationship._node_1 = verb_node_1
            relationship._node_2 = verb_node_2
            relationship._type = "cooccurrence"
            relationships.append(relationship)
    
for relationship in relationships:
    print(relationship)