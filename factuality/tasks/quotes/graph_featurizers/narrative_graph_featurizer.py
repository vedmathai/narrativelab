from narrativity.graph_generator.dependency_parse_pipeline.parser import NarrativeGraphGenerator
import torch
from transformers import RobertaTokenizer, RobertaModel
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

type_i2index = {
    "action_node": 0,
    "actor_relationship": 1,
    "entity_node": 2,
    "narrative_node": 3,
    "object_relationship": 4,
    "state_relationship": 5,
    "location_relationship": 6,
    "subject_relationship": 7,
    "prep_relationship": 8,
    "anecdotal_relationship": 9,
    "contradictory_relationship": 10,
    "absolute_temporal_node": 11,
    "absolute_temporal_relationship": 12,
    "causal_relationship": 13,
    "temporal_event_relationship": 14,
    "and_like_relationship": 15,
    "descriptor_relationship": 16,
}


class NarrativeGraphFeaturizer():
    def __init__(self):
        self._ngg = NarrativeGraphGenerator()
    
    def load(self):
        self._ngg.load()
        self._tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self._model = RobertaModel.from_pretrained('roberta-base').to(device)
        modules = [self._model.embeddings, *self._model.encoder.layer[:-2]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
        
    def featurize(self, sentence):
        sentence = re.sub("@ @ @ @ @ @ @", '', sentence)
        sentence = re.sub("` `", '"', sentence)
        graph = self._ngg.generate(sentence)
        node_id2index = {}
        adjacency = []
        types = set()
        x = []
        max_children = 0
        main_narrative_index = 0
        node_i = 0
        for node_i, node in enumerate(graph.nodes()):
            node_id2index[node.id()] = node_i
            types.add(node.type())
            if node.type() == 'narrative_node' and node._token is not None:
                if len(node._token.all_children()) > max_children:
                    max_children = len(node._token.all_children())
                    main_narrative_index = node_i
        for rel_i, rel in enumerate(graph.relationships(), node_i+1):
            node_id2index[rel.id()] = rel_i
            types.add(rel.type())
        X = [None for i in node_id2index]
        for type_i, _type in enumerate(sorted(list(types))):
            type_i2index[_type]
        for rel in graph.relationships():
            nodes = rel.nodes()
            for node0i, node0 in enumerate(nodes):
                for node1i, node1 in enumerate(nodes[node0i + 1:]):
                    if node0 is None or node1 is None:
                        continue
                    x = node_id2index[node0.id()]
                    y = node_id2index[node1.id()]
                    rel_idx = node_id2index[rel.id()]
                    first_node_type = node0.type()
                    second_node_type = node1.type()
                    rel_type = rel.type()
                    first_node_type_encoding = [0] * len(type_i2index.keys())
                    second_node_type_encoding = [0] * len(type_i2index.keys())
                    rel_type_encoding = [0] * len(type_i2index.keys())
                    first_node_type_encoding[type_i2index[first_node_type]] = 1
                    second_node_type_encoding[type_i2index[second_node_type]] = 1
                    rel_type_encoding[type_i2index[rel_type]] = 1

                    first_node_type_encoding = torch.Tensor(first_node_type_encoding).to(device)
                    second_node_type_encoding = torch.Tensor(second_node_type_encoding).to(device)
                    rel_type_encoding = torch.Tensor(rel_type_encoding).to(device)
                    adjacency.extend([[x, rel_idx], [rel_idx, y]])
                    pooled_node_1 = self.node2pooled(node0)
                    pooled_node_2 = self.node2pooled(node1)
                    pooled_rel = self.node2pooled(rel)
                    pooled_node_1 = torch.cat([pooled_node_1[0], first_node_type_encoding], 0)
                    pooled_node_2 = torch.cat([pooled_node_2[0], second_node_type_encoding], 0)
                    pooled_rel = torch.cat([pooled_rel[0], rel_type_encoding], 0)
                    X[x] = pooled_node_1
                    X[y] = pooled_node_2
                    X[rel_idx] = pooled_rel
        for ii, i in enumerate(X):
            if i is None:
                X[ii] = torch.tensor([0.0] * (768 + len(type_i2index))).to(device)
        x = [i[0] for i in adjacency]
        y = [i[1] for i in adjacency]
        if len(X) == 0:
            return
        X = torch.stack(X).to(device)
        edge_index = torch.tensor([x, y], dtype=int).to(device)
        return X, edge_index, main_narrative_index

    def node2pooled(self, node):
        display_name = node.display_name()
        if display_name is None or display_name.strip() == '':
            display_name = '[PAD]'
        text = ' '.join(display_name.split())
        inputs = self._tokenizer([text], return_tensors="pt", padding='longest', max_length=1000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        pooled_output = outputs.pooler_output
        return pooled_output


if __name__ == '__main__':
    ngf = NarrativeGraphFeaturizer()
    ngf.load()
    ngf.featurize(
        """
        Foxes are small to medium-sized, omnivorous mammals belonging to several genera of the family Canidae. They have a flattened skull, upright, triangular ears, a pointed, slightly upturned snout, and a long bushy tail ("brush").

        Twelve species belong to the monophyletic "true fox" group of genus Vulpes. Approximately another 25 current or extinct species are always or sometimes called foxes; these foxes are either part of the paraphyletic group of the South American foxes, or of the outlying group, which consists of the bat-eared fox, gray fox, and island fox.[1]

        Foxes live on every continent except Antarctica. The most common and widespread species of fox is the red fox (Vulpes vulpes) with about 47 recognized subspecies.[2] The global distribution of foxes, together with their widespread reputation for cunning, has contributed to their prominence in popular culture and folklore in many societies around the world. The hunting of foxes with packs of hounds, long an established pursuit in Europe, especially in the British Isles, was exported by European settlers to various parts of the New World. 
        """)
