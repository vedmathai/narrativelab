from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
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
    'questionable_source': 'right',
    'conspiracy_pseudoscience': 'right',
}

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



with open('factuality/data/quotes/creator/quotes.csv') as f:
    reader = csv.reader(f, delimiter='\t')
    documents = []
    y = []
    doc_list = list(reader)
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

    def featurize(doc_list):
        featurized = []
        labels = []
        for ri, r in enumerate(doc_list):
            label = r[1]
            featurized.append(r[0])
            labels.append(r[1])
            
            #y.append()
            #l = label_mapping[r[1][:-4]] + r[1][-4:]
            #label2examples[l].append(r[0])
            graph = ngg.generate(r[0])
            for narrative1 in graph.narrative_nodes().values():
                narrative = narrative1
            """
                #if len(narrative1.anecdotal_out_relationships()) == 0:
                #    print(r)
                for anecdotal_relationship in narrative1.anecdotal_out_relationships():
                    narrative = anecdotal_relationship.narrative_2()
                    if narrative is not None:
                        
                        #print(narrative1.display_name())
                        n1 = ' '.join(narrative1.display_name().split('->'))
            """
            for rel in narrative.actor_relationships():
                for obj_rel in (narrative.indirect_object_relationships() + narrative.direct_object_relationships()):
                    actions = [i.display_name() for i in narrative.actions()]
                    dn = ' '.join([ rel.actor().display_name(), actions[0], obj_rel.object().display_name()])
                    featurized.append(dn)
                    labels.append(label)
            print(narrative.display_name())

        return featurized, labels

            

train_featurized, train_labels = featurize(train_doc_list)
test_featurized, test_labels = featurize(test_doc_list)
vectorizer = CountVectorizer(binary=True)
X_train = vectorizer.fit_transform(train_featurized).toarray()
X_test = vectorizer.transform(test_featurized).toarray()
#classifier = MultinomialNB()
classifier = RandomForestClassifier()
classifier.fit(X_train, train_labels) 

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)
y_pred = classifier.predict(X_test)
#y_pred = [i[-4:] for i in y_pred]
#y_test = [i[-4:] for i in y_test]



print(confusion_matrix(test_labels, y_pred))
print(classification_report(test_labels, y_pred))
print(accuracy_score(test_labels, y_pred))