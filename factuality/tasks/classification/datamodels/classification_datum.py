class ClassificationDatum:
    def __init__(self):
        self._label = None
        self._text = None

    def label(self):
        return self._label
    
    def set_label(self, label):
        self._label = label

    def text(self):
        return self._text
    
    def set_text(self, text):
        self._text = text

    def to_dict(self):
        return {
            'label': self.label(),
            'text': self.text(),
        }
    
    @staticmethod
    def from_dict(data_dict):
        datum = ClassificationDatum()
        datum.set_label(data_dict['label'])
        datum.set_text(data_dict['text'])
        return datum
