class ECBCrossDocCoref:
    def __init__(self):
        self._rid = None
        self._note = None
        self._sources = []
        self._target = []
    
    def rid(self):
        return self._rid
    
    def set_rid(self, rid):
        self._rid = rid

    def note(self):
        return self._note
    
    def set_note(self, note):
        self._note = note

    def sources(self):
        return self._sources
    
    def set_sources(self, sources):
        self._sources = sources

    def add_source(self, source):
        self._sources.append(source)

    def target(self):
        return self._target
    
    def set_target(self, target):
        self._target = target

    def add_target(self, target):
        self._target.append(target)

    @staticmethod
    def from_bs(bs):
        cross_doc_coref = ECBCrossDocCoref()
        cross_doc_coref.set_rid(bs.attrs['r_id'])
        cross_doc_coref.set_note(bs.attrs['note'])
        sources = bs.find_all('source')
        for source in sources:
            cross_doc_coref.add_source(source.attrs['m_id'])
        target = bs.find('target')
        cross_doc_coref.set_target(target.attrs['m_id'])
        return cross_doc_coref
        
    def to_dict(self):
        return {
            'rid': self.rid(),
            'note': self.note(),
            'sources': self.sources(),
            'target': self.target()
        }
