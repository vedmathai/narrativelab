class ECBInstanceMarkable:
    def __init__(self):
        self._markable_type = None
        self._mid = None
        self._related_to = None
        self._tag_descriptor = None
        self._instance_id = None
    
    def mid(self):
        return self._mid
    
    def markable_type(self):
        return self._markable_type
    
    def related_to(self):
        return self._related_to
    
    def tag_descriptor(self):
        return self._tag_descriptor
    
    def instance_id(self):
        return self._instance_id
    
    def set_mid(self, mid):
        self._mid = mid

    def set_markable_type(self, markable_type):
        self._markable_type = markable_type

    def set_related_to(self, related_to):
        self._related_to = related_to

    def set_tag_descriptor(self, tag_descriptor):
        self._tag_descriptor = tag_descriptor

    def set_instance_id(self, instance_id):
        self._instance_id = instance_id
    
    @staticmethod
    def from_bs(bs):
        markable = ECBInstanceMarkable()
        markable.set_mid(bs.attrs['m_id'])
        markable.set_markable_type(bs.name)
        markable.set_related_to(bs.attrs['related_to'])
        markable.set_tag_descriptor(bs.attrs['tag_descriptor'])
        markable.set_instance_id(bs.attrs['instance_id'])
        return markable

    def to_dict(self):
        return {
            'mid': self.mid(),
            'markable_type': self.markable_type(),
            'related_to': self.related_to(),
            'tag_descriptor': self.tag_descriptor(),
            'instance_id': self.instance_id()
        }
