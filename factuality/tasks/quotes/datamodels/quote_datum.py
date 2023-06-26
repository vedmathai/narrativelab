class QuoteDatum:
    def __init__(self):
        self._text = None
        self._label = None

    def text(self):
        return self._text
    
    def label(self):
        return self._label
    
    def set_text(self, text):
        self._text = text

    def set_label(self, label):
        self._label = label

    @staticmethod
    def from_dict(val):
        quote_datum = QuoteDatum()
        quote_datum._text = val['text']
        quote_datum._label = val['label']
        return quote_datum
    
    def to_dict(self):
        return {
            'text': self.text(),
            'label': self.label()
        }
