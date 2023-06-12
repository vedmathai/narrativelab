import random
from collections import defaultdict
import csv
import numpy as np

from narrativity.graph_generator.dependency_parse_pipeline.parser import NarrativeGraphGenerator
ngg = NarrativeGraphGenerator()

ngg.load()


allowed_labels = [
    "left_center_bias-rep",
    "right_center_bias-rep",
    "left_center_bias-dem",
    "right_center_bias-dem",
]

label_mapping = {
    'left_center_bias': 'left',
    'right_center_bias': 'right',
    'left_bias': 'left',
    'right_bias': 'right',
    'questionable_source': 'right',
    'conspiracy_pseudoscience': 'right',
}
"""
with open('factuality/data/quotes/creator/non_quotes.csv') as f:
    reader = csv.reader(f, delimiter='\t')
    documents = []
    y = []
    for ri, r in enumerate(reader):
        if r[1][:-4] in label_mapping and ri % 4 == 0:
            label = label_mapping[r[1][:-4]] + r[1][-4:]
            documents.append((r[0], label))

"""

with open('factuality/data/quotes/creator/quotes.csv') as f:
    reader = csv.reader(f, delimiter='\t')
    documents = []
    y = []
    doc_list = list(reader)
    random.shuffle(doc_list)
    seen = set()
    for ri, r in enumerate(doc_list):
        if ri % 100 == 0:
            print(ri / 56000)
        if ri == 56000:
            break
        if r[0] in seen:
            continue

        seen.add(r[0])
        if r[1][:-4] in label_mapping and ri % 1 == 0:
            label = label_mapping[r[1][:-4]] + r[1][-4:]
            #documents.append((r[0], label))
            graph = ngg.generate(r[0])
            for narrative in graph.narrative_nodes().values():
                #for anecdotal_relationship in narrative1.anecdotal_out_relationships():
                #    narrative = anecdotal_relationship.narrative_2()
                    #if narrative is not None:
                        #print(label, anecdotal_relationship.narrative_1().display_name(), narrative.display_name(), r[0])

                for rel in narrative.actor_relationships():
                    for obj_rel in narrative.indirect_object_relationships() + narrative.direct_object_relationships():
                            actions = [i.display_name() for i in narrative.actions()]
                            dn = ' '.join([rel.actor().display_name(), actions[0], obj_rel.object().display_name()])
                            documents.append((dn, label))
                        #for state_rel in narrative.state_relationships():
                        #    state = state_rel.state()
                        #    # narrative = state_rel.narrative()
                        #    sub_rels = narrative.subject_relationships()
                        #    for sub_rel in sub_rels:
                        #        subject = sub_rel.subject()
                        #        d = ' '.join([subject.display_name(), state_rel.auxiliary(), state.display_name()])
                        #        documents.append((d, label))


pxy = defaultdict(lambda: defaultdict(int))
px = defaultdict(int)
py = defaultdict(int)

grams = 3
for d, label in documents:
    d = d.lower()
    #for i in range(len(d.split()) - (grams - 1)):
    #    bigram = ' '.join(d.split()[i: i+grams])
    pxy[label][d] += 1
    #for i in range(len(d.split()) - (grams - 1)):
    #    bigram = ' '.join(d.split()[i: i+grams])
    py[d] += 1
    px[label] += 1

counts = dict(py)

total = 0
for label in pxy:
    total += sum(pxy[label].values())
for label in pxy:
    for word in pxy[label]:
        pxy[label][word] /= total
total = sum(px.values())
for label in px:
    px[label] /= total

total = sum(py.values())
for word in py:
    py[word] /= total

score = defaultdict(lambda: defaultdict(int))
for label in pxy:
    for word in pxy[label]:
        score[label][word] = pxy[label][word] * np.log2(pxy[label][word] / (px[label] * py[word]))

for label in score:
    items = score[label].items()
    items = sorted(items, key=lambda x: x[1])
    print ()
    print(label, [(i[0], counts[i[0]]) for i in items if '@' not in i[0] and ''.join(i[0].split()).isalpha()][-60:])


