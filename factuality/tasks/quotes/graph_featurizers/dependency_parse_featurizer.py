from narrativity.graph_generator.dependency_parse_pipeline.parser import NarrativeGraphGenerator
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
import re
import uuid

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
    "entity": 0,
    "dep": 1,
    "relcl": 2,
    "poss": 3,
    "det": 4,
    "nmod": 5,
    "nsubj": 6,
    "nsubjpass": 7,
    "ccomp": 8,
    "appos": 9,
    "cc": 10,
    "compound": 11,
    "oprd": 12,
    "advmod": 13,
    "attr": 14,
    "conj": 15,
    "npadvmod": 16,
    "aux": 17,
    "agent": 18,
    "dobj": 19,
    "prep": 20,
    "auxpass": 21,
    "amod": 22,
    "punct": 23,
    "pobj": 24,
    "acl": 25,
    "nummod": 26,
    "dative": 27,
    "xcomp": 28,
    "advcl": 29,
    "mark": 30,
    "case": 31,
    "acomp": 32,
    "neg": 33,
    "prt": 34,
    "pcomp": 35,
    "csubj": 36,
    "quantmod": 37,
    "meta": 38,
    "preconj": 39,
    "expl": 40,
    "others": 41,
    "cooccurrence": 42,
}




class DependencyParseFeaturizer():
    def __init__(self):
        self._corpus2spacy = Corpus2spacy()

    
    def load(self, device):
        self._device = device
        self._corpus2spacy.load()
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
        node_i = 0
        token_i2node = {}
        nodes = []
        roots = []
        relationships = []
        for sentence in spacy_corpus.sentences():
            for token in sentence.tokens():
                node = Node()
                node._token = token
                node._text = token.text()
                node_id2index[node._id] = len(node_id2index)
                token_i2node[token.i()] = node
                if len(token.all_children()) > max_children:
                    max_children = len(token.all_children())
                    main_narrative_index = node_id2index[node._id]
                nodes.append(node)
                if token.pos() == 'ROOT':
                    roots.append(node)
        for sentence in spacy_corpus.sentences():
            for token in sentence.tokens():
                for dep, children in token.children().items():
                    for child in children:
                        first_node = token_i2node[token.i()]
                        second_node = token_i2node[child.i()]
                        relationship = Relationship()
                        node_id2index[relationship._id] = len(node_id2index)
                        relationship._type = dep
                        relationship._node_1 = first_node
                        relationship._node_2 = second_node
                        nodes.append(relationship)
                        types.add(dep)
                        relationships.append(relationship)
        for root_1 in roots:
            for root_2 in roots:
                if root_1 != root_2:
                    relationship = Relationship()
                    node_id2index[relationship._id] = len(node_id2index)
                    relationship._type = 'cooccurrence'
                    relationship._node_1 = root_1
                    relationship._node_2 = root_2
                    relationships.append(relationship)
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
            rel_type_encoding[type_i2index.get(rel_type, 41)] = 1

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
    ngf = DependencyParseFeaturizer()
    ngf.load('cpu')
    ngf.featurize(
        """
        Foxes are small to medium-sized, omnivorous mammals belonging to several genera of the family Canidae. They have a flattened skull, upright, triangular ears, a pointed, slightly upturned snout, and a long bushy tail ("brush").

        Twelve species belong to the monophyletic "true fox" group of genus Vulpes. Approximately another 25 current or extinct species are always or sometimes called foxes; these foxes are either part of the paraphyletic group of the South American foxes, or of the outlying group, which consists of the bat-eared fox, gray fox, and island fox.[1]

        Foxes live on every continent except Antarctica. The most common and widespread species of fox is the red fox (Vulpes vulpes) with about 47 recognized subspecies.[2] The global distribution of foxes, together with their widespread reputation for cunning, has contributed to their prominence in popular culture and folklore in many societies around the world. The hunting of foxes with packs of hounds, long an established pursuit in Europe, especially in the British Isles, was exported by European settlers to various parts of the New World. 
        """)
