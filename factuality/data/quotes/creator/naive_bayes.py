from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import re

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random
import csv

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
}

label_mapping2 = {
    'left_bias': 'left',
    'right_bias': 'right',
    'questionable_source': 'right',
    'conspiracy_pseudoscience': 'right',
}

subject_object_counters = defaultdict(int)

topics = ['Trump', 'Biden', 'Democrats', 'Republicans', 'democrat', 'republican', 'democratic', 'China', 'Covid', 'Virus', 'clinton', 'obama', 'pelosi', 'pence', 'Giuliani', 'kamala', 'harris', 'sanders', 'floyd', 'clinton', 'bernie', 'court', 'police']


anecdotal_verbs = ["observe", "observes", "observed", "describe", "describes", "described", "discuss", "discusses", "discussed",
    "report", "reports", "reported", "outline", "outlines", "outlined", "remark", "remarks", "remarked", 	
    "state", "states", "stated", "go on to say that", "goes on to say that", "went on to say that", 	
    "quote that", "quotes that", "quoted that", "say", "says", "said", "mention", "mentions", "mentioned",
    "articulate", "articulates", "articulated", "write", "writes", "wrote", "relate", "relates", "related",
    "convey", "conveys", "conveyed", "recognise", "recognises", "recognised", "clarify", "clarifies", "clarified",
    "acknowledge", "acknowledges", "acknowledged", "concede", "concedes", "conceded", "accept", "accepts", "accepted",
    "refute", "refutes", "refuted", "uncover", "uncovers", "uncovered", "admit", "admits", "admitted",
    "demonstrate", "demonstrates", "demonstrated", "highlight", "highlights", "highlighted", "illuminate", "illuminates", "illuminated", 							  
    "support", "supports", "supported", "conclude", "concludes", "concluded", "elucidate", "elucidates", "elucidated",
    "reveal", "reveals", "revealed", "verify", "verifies", "verified", "argue", "argues", "argued", "reason", "reasons", "reasoned",
    "maintain", "maintains", "maintained", "contend", "contends", "contended", "hypothesise", "hypothesises", "hypothesised",
    "propose", "proposes", "proposed", "theorise", "theorises", "theorised", "feel", "feels", "felt", "consider", "considers", "considered", 						  
    "assert", "asserts", "asserted", "dispute", "disputes", "disputed", "advocate", "advocates", "advocated",
    "opine", "opines", "opined", "think", "thinks", "thought", "imply", "implies", "implied", "posit", "posits", "posited",
    "show", "shows", "showed", "illustrate", "illustrates", "illustrated", "point out", "points out", "pointed out",
    "prove", "proves", "proved", "find", "finds", "found", "explain", "explains", "explained", "agree", "agrees", "agreed",
    "confirm", "confirms", "confirmed", "identify", "identifies", "identified", "evidence", "evidences", "evidenced",
    "attest", "attests", "attested", "believe", "believes", "believed", "claim", "claims", "claimed", "justify", "justifies", "justified", 							  
    "insist", "insists", "insisted", "assume", "assumes", "assumed", "allege", "alleges", "alleged", "deny", "denies", "denied",
    "speculate", "speculates", "speculated", "disregard", "disregards", "disregarded", "suppose", "supposes", "supposed",
    "conjecture", "conjectures", "conjectured", "surmise", "surmises", "surmised", "note", "notes", "noted",
    "suggest", "suggests", "suggested", "challenge", "challenges", "challenged", "critique", "critiques", "critiqued",
    "emphasise", "emphasises", "emphasised", "declare", "declares", "declared", "indicate", "indicates", "indicated",
    "comment", "comments", "commented", "uphold", "upholds", "upheld", "perceives", "perceive", "perceived"
]


 
"""
with open('factuality/data/quotes/creator/non_quotes.csv') as f:
    reader = csv.reader(f, delimiter='\t')
    documents = []
    y = []
    for ri, r in enumerate(reader):
        if r[1][:-4] in label_mapping and ri % 4 == 0:
            documents.append(r[0])
            #y.append(label_mapping[r[1][:-4]] + r[1][-4:])
            y.append(r[1][-4:])
            #y.append(label_mapping[r[1][:-4]])
            #y.append(random.choice(allowed_labels))

tfidfconverter = CountVectorizer()
"""


with open('factuality/data/quotes/creator/masked_quotes.csv') as f:
    reader = csv.reader(f, delimiter='\t')
    documents = []
    y = []
    doc_list = list(reader)
    random.seed(22)
    random.shuffle(doc_list)
    label2docs = defaultdict(list)
    seen = set()
    for ri, r in enumerate(doc_list):
        if ri % 1 != 0 or r[0] in seen or r[1][:-4] not in label_mapping:
            continue
        seen.add(r[0])
        label = label_mapping[r[1][:-4]] + r[1][-4:]
        label2docs[label].append(r[0])
    train_doc_list = []
    test_doc_list = []
    min_size = min([len(i) for i in label2docs.values()])
    train_size = int(0.7 * min_size)
    for k in label2docs:
        for i in label2docs[k][:train_size]:
            train_doc_list.append((i, k))
    for k in label2docs:
        for i in label2docs[k][train_size:min_size]:
            test_doc_list.append((i, k))

def featurize(doc_list, train):
    featurized = []
    labels = []
    for ri, r in enumerate(doc_list):
        label = r[1]
        #featurized.append(r[0])
        #labels.append(r[1])
        
        #y.append()
        #l = label_mapping[r[1][:-4]] + r[1][-4:]
        #label2examples[l].append(r[0])
        augmented_r0 = r[0]
        if True:
            annecdotal_narratives = set()
            graph = ngg.generate(r[0])
            dn = ""
            quote = ''
            quote_dep = ''
            for narrative1 in graph.narrative_nodes().values():
                if 'MASK' in narrative1.display_name():
                    ar = narrative1.anecdotal_out_relationships()
                    for rel in ar:
                        narrative2 = rel.narrative_2()
                        if narrative2 is not None:
                            annecdotal_narratives.add(narrative2.id())
                            pipe = [narrative2]
                            while len(pipe) > 0:
                                top = pipe.pop(0)
                                relationships = sum([
                                        top.anecdotal_out_relationships(),
                                        top.prep_out_relationships(),
                                        top.causal_out_relationships(),
                                        top.contradictory_out_relationships(),
                                        top.and_like_relationships()
                                    ], [])
                                if len(relationships) > 0:
                                    for i in relationships:
                                        if i.narrative_2() is not None:
                                            pipe.append(i.narrative_2())
                                annecdotal_narratives.add(top)

            for narrative1 in graph.narrative_nodes().values():
                narrative = narrative1
                if narrative.id() not in annecdotal_narratives:
                    pass
                if narrative.is_state() is False:
                    children = []
                    pipe = [narrative._token]
                    while len(pipe) > 0 and pipe[0] is not None:
                        top = pipe.pop(0)
                        children.append(top)
                        pipe.extend(top.all_children())
                    for c in children:
                        subject_object_counters[c.text().lower()] += 1
                    quote += ' ' + ' '.join([i.text() for i in children])
                    quote_dep += ' ' + ' '.join(['{}_{}'.format(i.text(), i.dep()) for i in children])
                    quote_dep = quote_dep + ' ' + quote
                    dn = str(quote)
                    triple = narrative.display_name()
                    actor, action, object = triple.split('->')
                    actor = ' '.join(actor.split('|'))
                    object = ' '.join(object.split('|'))
                    if action in anecdotal_verbs:
                        pass
                    actor = ' '.join('{}_actor'.format(i) for i in actor.split())
                    action = ' '.join('{}_action'.format(i) for i in action.split())
                    object = ' '.join('{}_object'.format(i) for i in object.split())

                    indirect_objects = narrative.indirect_object_relationships()
                    indirect_objects = ' '.join([obj_rel.object().display_name() for obj_rel in indirect_objects])
                    indirect_objects = ' '.join('{}_indirect'.format(i) for i in indirect_objects.split())
                    locations = narrative.location_relationships()
                    locations = ' '.join([loc_rel.location().display_name() for loc_rel in locations])
                    locations = ' '.join('{}_location'.format(i) for i in locations.split())
                    abs_time = narrative.absolute_temporal_relationships()
                    abs_time = ' '.join([abs_time_rel.absolute_temporal_node().display_name() for abs_time_rel in abs_time])
                    abs_time = ' '.join('{}_abs_time'.format(i) for i in locations.split())
                    #if narrative.id() in annecdotal_narratives:
                    #    dn += ' ' + ' '.join([object, indirect_objects, locations, abs_times])
                    #else:
                    dn += ' ' + ' '.join([actor, action, object, indirect_objects, locations, abs_time])
                    augmented_r0 += ' ' + ' '.join([actor, action, object, indirect_objects, locations, abs_time])


                if narrative.is_state() is True:
                    triple = narrative.display_name()
                    subject, aux, object = triple.split('->')
                    subject = ' '.join(subject.split('|'))
                    object = ' '.join(object.split('|'))

                    subject = ' '.join('{}_subject'.format(i) for i in subject.split())
                    aux = ' '.join('{}_aux'.format(i) for i in aux.split())
                    object = ' '.join('{}_object'.format(i) for i in object.split())
                    dn += ' ' + ' '.join([subject, aux, object])
                    augmented_r0 += ' ' + ' '.join([subject, aux, object])

            doc = dn
            if len(dn.strip()) > 0 and len(quote.strip()) > 0:# and any(i.lower() in quote.lower() for i in topics):
                if train is True:
                    #print(dn)
                    #print(quote)
                    #print('\n' * 4)
                    featurized.append(doc)
                    labels.append(label)
                else:
                    featurized.append(doc)
                    labels.append(label)
    return featurized, labels

          
#doc_list = [["For all those of you who voted for President Trump , I understand the disappointment tonight , '' MASK said", 'label']]
#random.shuffle(train_doc_list)
train_featurized, train_labels = featurize(train_doc_list, True)
test_featurized, test_labels = featurize(test_doc_list, False)
#featurize(doc_list, True)
#print(sorted(subject_object_counters.items(), key=lambda x: x[1]))
#featurize(doc_list, True)
vectorizer = CountVectorizer(binary=True)
X_train = vectorizer.fit_transform(train_featurized).toarray()
X_test = vectorizer.transform(test_featurized).toarray()
classifier = MultinomialNB()

classifier.fit(X_train, train_labels) 

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)
y_pred = classifier.predict(X_test)
#y_pred = [i[-4:] for i in y_pred]
#y_test = [i[-4:] for i in y_test]



print(confusion_matrix(test_labels, y_pred))
print(classification_report(test_labels, y_pred))
print(accuracy_score(test_labels, y_pred))

# The best is to add the full quote words, the suffixed subj-obj-verb and this condition if len(dn.strip()) > 0 and len(annecdotal_narratives) > 0 and any(i.lower() in dn.lower() for i in topics)