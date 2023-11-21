class ECBTokenMarkable:
    def __init__(self):
        self._mid = None
        self._markable_type = None
        self._token_anchors = []
    
    def markable_type(self):
        return self._markable_type
        
    def set_markable_type(self, markable_type):
        self._markable_type = markable_type

    def token_anchors(self):
        return self._token_anchors
    
    def set_token_anchors(self, token_anchors):
        self._token_anchors = token_anchors

    def add_token_anchor(self, token_anchor):
        self._token_anchors.append(token_anchor)

    def mid(self):
        return self._mid
    
    def set_mid(self, mid):
        self._mid = mid
    
    @staticmethod
    def from_bs(bs):
        markable = ECBTokenMarkable()
        tag = bs.name
        markable.set_markable_type(tag)
        markable.set_mid(bs.attrs['m_id'])
        token_anchors = bs.find_all('token_anchor')
        for token_anchor in token_anchors:
            markable.add_token_anchor(token_anchor.attrs['t_id'])
        return markable

    def to_dict(self):
        return {
            'mid': self.mid(),
            'markable_type': self.markable_type(),
            'token_anchors': self.token_anchors()
        }
