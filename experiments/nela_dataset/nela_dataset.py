import os
import json
import csv
from collections import defaultdict

source2leaning = {}
with open('/home/lalady6977/oerc/projects/narrative/narrativelab/experiments/nela_dataset/mapping.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for r in reader:
        source = ''.join(r[0].lower().split())
        source2leaning[source] = r[1]
print(len(source2leaning.keys()))

said_verbs = ["observe", "observes", "observed", "describe", "describes", "described", "discuss", "discusses", "discussed",
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
    "comment", "comments", "commented", "uphold", "upholds", "upheld"
]
said_verbs = ['said']

class main():
    def __init__(self):
        pass

    def read_dataset(self):
        topic_counter = defaultdict(int)
        leaning_counter = defaultdict(int)
        leaning2content = defaultdict(list)
        dataset_folder = "/home/lalady6977/oerc/projects/data/nela-elections-2020.json/nela-elections-2020/newsdata"
        filenames = os.listdir(dataset_folder)
        for filename in filenames:
            fname = filename[:-5]
            filepath = os.path.join(dataset_folder, filename)
            with open(filepath) as f:
                data = json.load(f)
                for i in data:
                    content = i['content'].split()
                    for said_verb in said_verbs:
                        search_term = "CNN {}".format(said_verb)
                        if search_term in i['content']:
                            point = i['content'].find(search_term)
                            j = point
                            k = point
                            while j > 0 and i['content'][j] != '.':
                                j -= 1
                            while k < len(i['content']) and i['content'][k] != '.':
                                k += 1
                            leaning_counter[source2leaning.get(fname)] += 1
                            leaning2content[source2leaning.get(fname)].append(i['content'][j + 1: k])
                            #if source2leaning.get(fname) == 'right_bias':
                            #    print(' '.join(i['content'][j + 1: k].split()), source2leaning.get(fname))

                    for k in range(len(content) - 1):
                        bigram = ' '.join([content[k], content[k+1]])
                        if bigram.istitle() and ''.join(bigram.split()).isalpha() and bigram + ' said' in i['content']:
                            topic_counter[bigram] += 1

        for i in sorted(topic_counter.items(), key=lambda i: i[1])[-1000:]:
            print(i[0])

        with open('/home/lalady6977/oerc/projects/narrative/narrativelab/experiments/nela_dataset/cnn.csv', 'wt') as f:
            writer = csv.writer(f, delimiter='\t')
            for k in leaning2content:
                for c in leaning2content[k]:
                    writer.writerow([k, c]) 
        
        print(leaning_counter)


if __name__ == '__main__':
    main().read_dataset()