
class VitaminCDatum:
    def __init__(self):
        self._unique_id = None
        self._case_id = None
        self._wiki_revision_id = None
        self._fever_id = None
        self._label = None
        self._claim = None
        self._evidence = None
        self._page = None
        self._revision_type = None

    def unique_id(self):
        return self._unique_id

    def case_id(self):
        return self._case_id
    
    def wiki_revision_id(self):
        return self._wiki_revision_id
    
    def fever_id(self):
        return self._fever_id
    
    def label(self):
        return self._label
    
    def claim(self):
        return self._claim
    
    def evidence(self):
        return self._evidence
    
    def page(self):
        return self._page
    
    def revision_type(self):
        return self._revision_type
    
    def set_unique_id(self, unique_id):
        self._unique_id = unique_id

    def set_case_id(self, case_id):
        self._case_id = case_id

    def set_wiki_revision_id(self, wiki_revision_id):
        self._wiki_revision_id = wiki_revision_id

    def set_fever_revision_id(self, fever_revision_id):
        self._fever_revision_id = fever_revision_id

    def set_label(self, label):
        self._label = label

    def set_claim(self, claim):
        self._claim = claim

    def set_evidence(self, evidence):
        self._evidence = evidence

    def set_page(self, page):
        self._page = page

    def set_revision_type(self, type_id):
        self._type_id = type_id

    @staticmethod
    def from_dict(val):
        datum = VitaminCDatum()
        datum.set_unique_id(val['unique_id'])
        datum.set_case_id(val['case_id'])
        datum.set_wiki_revision_id(val.get('wiki_revision_id'))
        datum.set_fever_revision_id(val.get('fever_revision_id'))
        datum.set_label(val['label'])
        datum.set_claim(val['claim'])
        datum.set_evidence(val['evidence'])
        datum.set_page(val['page'])
        datum.set_revision_type(val['revision_type'])
        return datum
    
    def to_dict(self):
        return {
            "unique_id": self.unique_id,
            "case_id": self.case_id(),
            "wiki_revision_id": self.wiki_revision_id(),
            "label": self.label(),
            "claim": self.claim(),
            "evidence": self.evidence(),
            "page": self.page(),
            "revision_type": self.revision_type(),
        }
