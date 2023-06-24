import csv
import re

from factuality.data.nela.datareader import NelaDatareader

persons_dict = {
    'Clinton': 'dem',
    'Trump': 'rep',
    'Biden': 'dem',
    'Pence': 'rep',
    'Schumer': 'dem',
    'Obama': 'dem',
    'Giuliani': 'rep',
    #'Bill Barr': 'rep',
    #'Lindsey Graham': 'rep',
    #'Mike Pompeo': 'rep',
    #'John Bolton': 'rep',
    #'Bernie Sanders': 'dem',
}

verb_list = ["observe", "observes", "observed", "describe", "describes", "described", "discuss", "discusses", "discussed",
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


class QuotesDataCreator:
    def __init__(self):
        self._nela_datareader = NelaDatareader()
    
    def create_dataset(self):
        sentences = []
        dataset_list = []
        dataset = self._nela_datareader.read_dataset()
        size = len(dataset.data())
        for ii, i in enumerate(dataset.data()):
            print(ii/size)
            if i.label() is not None and len(i.label().media_bias_fact_check_label()) > 0:
                for person, party in persons_dict.items():
                    for d in i.data():
                        label = i.label().media_bias_fact_check_label()
                        name = i.label().name()
                        sentences = self._check_if_usable(d, person)

                        if len(sentences) > 0 and len(label) > 0:
                            for k in sentences:
                                dataset_list.append([k.strip(), '{}-{}'.format(label, party)])
        with open('factuality/data/quotes/creator/masked_quotes.csv', 'wt') as f:
            writer = csv.writer(f, delimiter='\t')
            for datum in dataset_list:
                writer.writerow(datum)

    def _check_if_usable(self, d, person):
        sentences = []
        for verb in verb_list:
            search_term = person
            speech_search_terms = ['{} {}'.format(person, verb)]
            for search_term in [search_term]:
                #if search_term in d.content() and not any(i in d.content() for i in speech_search_terms):
                if any(i in d.content() for i in speech_search_terms):
                    location = d.content().index(speech_search_terms[0])
                    #location = d.content().index(search_term)
                    j = location
                    k = location
                    while j >= 0 and d.content()[j] != '.':
                        j -= 1
                    while k < len(d.content()) - 2 and d.content()[k] != '.':
                        k += 1                    
                    sentence = d.content()[j+1:k]
                    sentence = re.sub(search_term, '[MASK]', d.content()[j+1:k])
                    sentences = [sentence]
        return sentences
                    

if __name__ == '__main__':
    quotes_data_creator = QuotesDataCreator()
    dataset = quotes_data_creator.create_dataset()
