class NelaLabel:
    def __init__(self):
        self._name = None
        self._media_bias_fact_check_label = None

    def name(self):
        return self._name
    
    def media_bias_fact_check_label(self):
        return self._media_bias_fact_check_label
    
    def set_name(self, name):
        self._name = name

    def set_media_bias_fact_check_label(self, media_bias_fact_check_label):
        self._media_bias_fact_check_label = media_bias_fact_check_label

    @staticmethod
    def from_csv(val):
        nela_labels = NelaLabel()
        nela_labels.set_name(val[0])
        nela_labels.set_media_bias_fact_check_label(val[33])
        return nela_labels
    
    def to_dict(self):
        return {
            'name': self.name(),
            'media_bias_fact_check_label': self.media_bias_fact_check_label(),
        }
    
    @staticmethod
    def from_dict(labels):
        nela_labels = NelaLabel()
        nela_labels.set_name(labels['name'])
        nela_labels.set_media_bias_fact_check_label(labels['media_bias_fact_check_label'])
        return nela_labels