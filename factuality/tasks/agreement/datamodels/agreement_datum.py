class AgreementDatum:
    def __init__(self):
        self._text_1 = None
        self._text_2 = None
        self._label = None

    def text_1(self):
        return self._text_1
    
    def text_2(self):
        return self._text_2

    def label(self):
        return self._label
    
    def set_text_1(self, text_1):
        self._text_1 = text_1

    def set_text_2(self, text_2):
        self._text_2 = text_2

    def set_label(self, label):
        self._label = label

    @staticmethod
    def from_dict(val):
        agreement_datum = AgreementDatum()
        agreement_datum._text_1 = val['text_1']
        agreement_datum._text_2 = val['text_2']
        agreement_datum._label = val['label']
        return agreement_datum
    
    def to_dict(self):
        return {
            'text_1': self.text_1(),
            'text_2': self.text_2(),
            'label': self.label()
        }
