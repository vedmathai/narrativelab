from narrativity.graph_generator.dependency_parse_pipeline.parser import NarrativeGraphGenerator
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import re
import uuid
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from collections import defaultdict
from jadelogs import JadeLogger



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

type_i2index = {
    "entity": 0,
    "others": 1,
    "cooccurrence": 2,
    "ARG0": 3,
    "ARG1": 4,
    "ARG2": 5,
    "ARG3": 6,
    "ARG4": 7,
    'R-ARG1': 8,
    'ARGM-LOC': 9,
    'ARGM-TMP': 10,
    "ARGM-PRD": 11,
    'ARGM-MOD': 12,
    'ARGM-ADV': 13,
    'ARGM-CAU': 14,
    'ARGM-NEG': 15,
    'ARGM-DIR': 16,
    'ARGM-EXT': 17,
    'R-ARG0' : 18,
    "ARGM-PRP": 19,
    "ARGM-MNR": 20,
    "ARGM-DIS": 21,
    "ARGM-GOL": 22,
    "ARGM-PNC": 23,
    'ARGM-ADJ': 24,
    'R-ARGM-TMP': 25,
    "R-ARGM-MNR": 26,
    "C-ARG0": 27,
    "C-ARG1": 28,
    "ARGM-COM": 29,
    "R-ARGM-CAU": 31,
    "R-ARG2": 30,
    "R-ARGM-LOC": 32,
}


class SRLParseFeaturizer():
    def __init__(self):
        self._jade_file_manager = JadeLogger().file_manager
        model_path = self._jade_file_manager.data_filepath("structured-prediction-srl-bert.2020.12.15.tar.gz")
        self._predictor = Predictor.from_path(model_path)

    def load(self, device):
        self._device = device
        self._tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self._model = RobertaModel.from_pretrained('roberta-base').to(self._device)
        modules = [self._model.embeddings, *self._model.encoder.layer[:-2]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
        self.pool = nn.MaxPool1d(1, stride=1)
        
        
    def featurize(self, sentence):
        display_name = '[PAD]'
        text = ' '.join(display_name.split('->'))
        inputs = self._tokenizer([text], return_tensors="pt", padding='longest', max_length=1000)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        pad = outputs.pooler_output
        sentence = re.sub("@ @ @ @ @ @ @", '', sentence)
        sentence = re.sub("` `", '"', sentence)
        inputs = self._tokenizer([sentence], return_tensors="pt", padding='longest', max_length=1000)
        tokens = self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        token_i = len(tokens) - 1
        token2embedding = {}
        total = ''
        while token_i > 0:
            token = tokens[token_i]
            if token[0] != 'Ġ' and tokens[token_i - 1] != '<s>':
                total = token + total
            else:
                if token_i != 1:
                    total = token[1:] + total
                else:
                    total = token + total  
                token2embedding[total] = outputs.last_hidden_state[0][token_i]
                total = ''
            token_i -= 1

        node_id2index = {}
        adjacency = []
        types = set()
        x = []
        main_narrative_index = 0
        nodes = []
        relationships = []

        string2node = {}
        srl_dict = self._predictor.predict(sentence=sentence)
        words = srl_dict['words']
        relationships = []
        previous_tag = ""
        verb_nodes = []
        for verb in srl_dict['verbs']:
            current_index = []
            tag2nodes = defaultdict(list)
            for tag_i, tag in  enumerate(verb['tags']):
                if tag[0] == 'B':
                    if len(current_index) > 0:
                        string = ' '.join(words[i] for i in current_index)
                        if string not in string2node:
                            node = Node()
                            node_id2index[node._id] = len(node_id2index)
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
                            node_id2index[node._id] = len(node_id2index)
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
                            node_id2index[relationship._id] = len(node_id2index)
                            relationship._node_1 = verb_node[0]
                            relationship._node_2 = node
                            relationship._type = tag
                            types.add(tag)
                            relationships.append(relationship)

        for verb_node_1_i, verb_node_1 in enumerate(verb_nodes):
            for verb_node_2_i, verb_node_2 in enumerate(verb_nodes):
                if verb_node_1_i < verb_node_2_i:
                    relationship = Relationship()
                    node_id2index[relationship._id] = len(node_id2index)
                    relationship._node_1 = verb_node_1
                    relationship._node_2 = verb_node_2
                    relationship._type = "cooccurrence"
                    relationships.append(relationship)

        main_narrative_index = node_id2index[verb_nodes[0]._id]
        X = [None for i in node_id2index]
        for rel in relationships:
            node0 = rel._node_1
            node1 = rel._node_2
            x = node_id2index[node0._id]
            y = node_id2index[node1._id]
            rel_idx = node_id2index[rel._id]
            first_node_type = node0._type
            second_node_type = node1._type
            rel_type = rel._type
            first_node_type_encoding = [0] * len(type_i2index.keys())
            second_node_type_encoding = [0] * len(type_i2index.keys())
            rel_type_encoding = [0] * len(type_i2index.keys())
            first_node_type_encoding[type_i2index[first_node_type]] = 1
            second_node_type_encoding[type_i2index[second_node_type]] = 1
            rel_type_encoding[type_i2index.get(rel_type, 1)] = 1


            first_node_type_encoding = torch.Tensor(first_node_type_encoding).to(self._device)
            second_node_type_encoding = torch.Tensor(second_node_type_encoding).to(self._device)
            rel_type_encoding = torch.Tensor(rel_type_encoding).to(self._device)
            adjacency.extend([[x, rel_idx], [rel_idx, y]])
            pooled_node_1 = self.node2pooled(node0, token2embedding, pad)
            pooled_node_2 = self.node2pooled(node1, token2embedding, pad)
            pooled_rel = self.node2pooled(rel, token2embedding, pad)
            pooled_node_1 = torch.cat([pooled_node_1[0], first_node_type_encoding], 0)
            pooled_node_2 = torch.cat([pooled_node_2[0], second_node_type_encoding], 0)
            pooled_rel = torch.cat([pooled_rel[0], rel_type_encoding], 0)
            X[x] = pooled_node_1
            X[y] = pooled_node_2
            X[rel_idx] = pooled_rel
        for ii, i in enumerate(X):
            if i is None:
                X[ii] = torch.tensor([0.0] * (768 + len(type_i2index))).to(self._device)
        x = [i[0] for i in adjacency]
        y = [i[1] for i in adjacency]
        if len(X) == 0:
            return
        X = torch.stack(X).to(self._device)
        edge_index = torch.tensor([x, y], dtype=int).to(self._device)
        return X, edge_index, main_narrative_index

    def node2pooled(self, node, token2embedding, pad):
        display_name = node._text
        if display_name is None or display_name.strip() == '' or ''.join(display_name.split('->')) == '':
            pooled_output = pad
        else:
            text = ' '.join(display_name.split('->'))
            text = text.split()
            array = []
            for t in text:
                array.append(token2embedding.get(t, pad[0]))
            tens = torch.stack(array)
            pooled_output = self.pool(tens)

        return pooled_output


if __name__ == '__main__':
    ngf = SRLParseFeaturizer()
    ngf.load('cpu')
    ngf.featurize(
        """
        Foxes are small to medium-sized, omnivorous mammals belonging to several genera of the family Canidae. They have a flattened skull, upright, triangular ears, a pointed, slightly upturned snout, and a long bushy tail ("brush").

        Twelve species belong to the monophyletic "true fox" group of genus Vulpes. Approximately another 25 current or extinct species are always or sometimes called foxes; these foxes are either part of the paraphyletic group of the South American foxes, or of the outlying group, which consists of the bat-eared fox, gray fox, and island fox.[1]

        Foxes live on every continent except Antarctica. The most common and widespread species of fox is the red fox (Vulpes vulpes) with about 47 recognized subspecies.[2] The global distribution of foxes, together with their widespread reputation for cunning, has contributed to their prominence in popular culture and folklore in many societies around the world. The hunting of foxes with packs of hounds, long an established pursuit in Europe, especially in the British Isles, was exported by European settlers to various parts of the New World. 
        """)
