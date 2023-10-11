import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import re
import uuid
import amrlib
import spacy
import penman
from collections import defaultdict

from narrativity.graph_generator.dependency_parse_pipeline.corpus2spacy import Corpus2spacy 

class Node:
    def __init__(self):
        self._id = str(uuid.uuid4())
        self._type = 'entity'
        self._token = None
        self._text = None
        self._relationships = []

class Relationship:
    def __init__(self):
        self._id = str(uuid.uuid4())
        self._text = ''
        self._type = None
        self._node_1 = None
        self._node_2 = None


type_i2index = {
    ":domain": 0,
    "entity": 1,
    ":snt2": 2,
    ":part": 3,
    ":ARG2": 4,
    ":poss": 5,
    ":op4": 6,
    ":location": 7,
    ":frequency": 8,
    ":ARG5": 9,
    ":instrument": 10,
    ":consist": 11,
    ":ARG3": 12,
    ":snt1": 13,
    ":ARG0": 14,
    ":accompanier": 15,
    ":ARG1": 16,
    ":op1": 17,
    ":quant": 18,
    "cooccurrence": 19,
    ":op3": 20,
    ":time": 21,
    ":manner": 22,
    ":op2": 23,
    ":name": 24,
    ":mod": 25,
    ":degree": 26,
}

class AMRParseFeaturizer():
    def __init__(self):
        self._corpus2spacy = Corpus2spacy()

    
    def load(self, device):
        self._device = device
        self._corpus2spacy.load()
        gtos = amrlib.load_gtos_model()
        amrlib.setup_spacy_extension()
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
        spacy_corpus = self._corpus2spacy.convert(sentence)
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
        max_children = 0
        main_narrative_index = 0
        relationships = []
        nodes, relationships, node_id2index = self.amr2nodes(sentence)
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
            rel_type_encoding[type_i2index[rel_type]] = 1

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
    
    def amr2nodes(self, text):
        nlp = spacy.load('en_core_web_lg')
        doc = nlp(text)
        graphs = doc._.to_amr()
        node2index = {}
        nodes = []
        node_id2node = {}
        head_nodes = []
        relationships = []
        overall_relationship_diff = defaultdict(int)
        out_relationships = defaultdict(int)
        for graph in graphs:
            g = penman.decode(graph)
            node_map = {}
            in_graph_relationship_diff = defaultdict(int)
            for instance in g.instances():
                if instance.target not in node_map:
                    node_map[instance.target] = Node()
                node = node_map[instance.target]
                node2index[node._id] = len(node2index)
                node_id2node[node._id] = node
                node._text = instance.target.split('-')[0]
                node_map[instance.source] = node
                nodes.append(node)
            for node in nodes:
                overall_relationship_diff[node._id] = 0
                in_graph_relationship_diff[node._id] = 0
            for edge in g.edges():
                relationship = Relationship()
                node2index[relationship._id] = len(node2index)
                relationship._type = edge.role
                relationship._node_1 = node_map[edge.source]
                relationship._node_2 = node_map[edge.target]
                overall_relationship_diff[relationship._node_1._id] += 1
                overall_relationship_diff[relationship._node_2._id] -= 1
                in_graph_relationship_diff[relationship._node_1._id] += 1
                in_graph_relationship_diff[relationship._node_2._id] -= 1
                relationship._node_1._relationships.append(relationship)
                relationship._node_2._relationships.append(relationship)
                relationships.append(relationship)
            max_diff = 0
            for node in in_graph_relationship_diff:
                if in_graph_relationship_diff[node] > max_diff:
                    max_diff = in_graph_relationship_diff[node]
                    head_node = node_id2node[node]
            head_nodes.append(head_node)
        for node_i in range(len(head_nodes) - 1):
            head_node = head_nodes[node_i]
            next_head_node = head_nodes[node_i + 1]
            relationship = Relationship()
            node2index[relationship._id] = len(node2index)
            relationship._type = 'cooccurrence'
            relationship._node_1 = head_node
            relationship._node_2 = next_head_node
            relationship._node_1._relationships.append(relationship)
            relationship._node_2._relationships.append(relationship)
            relationships.append(relationship)
        return nodes, relationships, node2index


if __name__ == '__main__':
    ngf = AMRParseFeaturizer()
    ngf.load('cpu')
    text_1 = """
        Foxes are small to medium-sized, omnivorous mammals belonging to several genera of the family Canidae. They have a flattened skull, upright, triangular ears, a pointed, slightly upturned snout, and a long bushy tail ("brush").

        Twelve species belong to the monophyletic "true fox" group of genus Vulpes. Approximately another 25 current or extinct species are always or sometimes called foxes; these foxes are either part of the paraphyletic group of the South American foxes, or of the outlying group, which consists of the bat-eared fox, gray fox, and island fox.

        Foxes live on every continent except Antarctica. The most common and widespread species of fox is the red fox (Vulpes vulpes) with about 47 recognized subspecies. The global distribution of foxes, together with their widespread reputation for cunning, has contributed to their prominence in popular culture and folklore in many societies around the world. The hunting of foxes with packs of hounds, long an established pursuit in Europe, especially in the British Isles, was exported by European settlers to various parts of the New World. 
    """

    text_2 = "Foxes are small to medium-sized, omnivorous mammals belonging to several genera of the family Canidae."
    ngf.featurize(text_1)
