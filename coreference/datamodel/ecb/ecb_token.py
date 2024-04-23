class ECBToken:
    def __init__(self):
        self._number = None
        self._sentence = None
        self._tid = None
        self._text = None

    def number(self):
        return self._number
    
    def sentence(self):
        return self._sentence
    
    def tid(self):
        return self._tid
    
    def text(self):
        return self._text
    
    def set_number(self, number):
        self._number = number

    def set_sentence(self, sentence):
        self._sentence = sentence

    def set_tid(self, tid):
        self._tid = tid

    def set_text(self, text):
        self._text = text

    @staticmethod
    def from_bs(bs):
        token = ECBToken()
        token._number = bs['number']
        token._sentence = bs['sentence']
        token._tid = bs['t_id']
        token._text = bs.text
        return token

    def to_dict(self):
        return {
            'number': self.number(),
            'sentence': self.sentence(),
            't_id': self.t_id(),
            'text': self.text()
        }
