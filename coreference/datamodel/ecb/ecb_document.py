from coreference.datamodel.ecb.ecb_token import ECBToken
from coreference.datamodel.ecb.ecb_instance_markable import ECBInstanceMarkable
from coreference.datamodel.ecb.ecb_token_markable import ECBTokenMarkable
from coreference.datamodel.ecb.ecb_instance_markable import ECBInstanceMarkable
from coreference.datamodel.ecb.ecb_cross_doc_coref import ECBCrossDocCoref


class ECBDocument:
    def __init__(self):
        self._tokens = []
        self._token_markables = []
        self._instance_markables = []
        self._cross_doc_corefs = []
    
    def tokens(self):
        return self._tokens
    
    def instance_markables(self):
        return self._instance_markables
    
    def token_markables(self):
        return self._token_markables
    
    def set_tokens(self, tokens):
        self._tokens = tokens

    def set_markables(self, markables):
        self._markables = markables

    def add_token(self, token):
        self._tokens.append(token)
    
    def add_token_markable(self, markable):
        self._token_markables.append(markable)

    def add_instance_markable(self, markable):
        self._instance_markables.append(markable)

    def cross_doc_corefs(self):
        return self._cross_doc_corefs
    
    def set_cross_doc_corefs(self, cross_doc_corefs):
        self._cross_doc_corefs = cross_doc_corefs

    def add_cross_doc_coref(self, cross_doc_coref):
        self._cross_doc_corefs.append(cross_doc_coref)

    @staticmethod
    def from_bs(bs):
        document = ECBDocument()
        tokens = bs.find_all('token')
        for token in tokens:
            token = ECBToken.from_bs(token)
            document.add_token(token)
        
        markables = bs.find('Markables')
        for child in markables.findChildren(recursive=False):
            if 'instance_id' in child.attrs:
                markable = ECBInstanceMarkable.from_bs(child)
                document.add_instance_markable(markable)
            else:
                markable = ECBTokenMarkable.from_bs(child)
                document.add_token_markable(markable)
        for child in bs.find_all('CROSS_DOC_COREF'):
            cross_doc_coref = ECBCrossDocCoref.from_bs(child)
            document.add_cross_doc_coref(cross_doc_coref)
        return document
        
    def to_dict(self):
        return {
            'tokens': [token.to_dict() for token in self.tokens()],
            'markables': [markable.to_dict() for markable in self.markables()],
            'cross_doc_corefs': [cross_doc_coref.to_dict() for cross_doc_coref in self.cross_doc_corefs()]
        }
