class CoreferenceLLMDatum:
    def __init__(self):
        self._text = None
        self._event_1 = None
        self._event_2 = None
        self._event_1_start = None
        self._event_1_end = None
        self._event_1_start = None
        self._event_1_start = None
        self._sentence_1 = None
        self._sentence_2 = None
        self._sentence_1_index = None
        self._sentence_2_index = None
        self._label = None

    def text(self):
        return self._text
    
    def event_1(self):
        return self._event_1
    
    def event_2(self):
        return self._event_2
    
    def event_1_start(self):
        return self._event_1_start
    
    def event_1_end(self):
        return self._event_1_end
    
    def event_2_start(self):
        return self._event_2_start
    
    def event_2_end(self):
        return self._event_2_end
    
    def sentence_1(self):
        return self._sentence_1
    
    def sentence_2(self):
        return self._sentence_2
    
    def sentence_1_index(self):
        return self._sentence_1_index
    
    def sentence_2_index(self):
        return self._sentence_2_index
    
    def label(self):
        return self._label
    
    def set_text(self, text):
        self._text = text

    def set_event_1(self, event_1):
        self._event_1 = event_1

    def set_event_2(self, event_2):
        self._event_2 = event_2

    def set_event_1_start(self, event_1_start):
        self._event_1_start = event_1_start

    def set_event_1_end(self, event_1_end):
        self._event_1_end = event_1_end

    def set_event_2_start(self, event_2_start):
        self._event_2_start = event_2_start

    def set_event_2_end(self, event_2_end):
        self._event_2_end = event_2_end

    def set_sentence_1(self, sentence_1):
        self._sentence_1 = sentence_1

    def set_sentence_2(self, sentence_2):
        self._sentence_2 = sentence_2

    def set_sentence_1_index(self, sentence_1_index):
        self._sentence_1_index = sentence_1_index

    def set_sentence_2_index(self, sentence_2_index):
        self._sentence_2_index = sentence_2_index

    def set_label(self, label):
        self._label = label

    @staticmethod
    def from_dict(val):
        datum = CoreferenceLLMDatum()
        datum.set_text(val['text'])
        datum.set_event_1(val['event_1'])
        datum.set_event_2(val['event_2'])
        datum.set_event_1_start(val['event_1_start'])
        datum.set_event_1_end(val['event_1_end'])
        datum.set_event_2_start(val['event_2_start'])
        datum.set_event_2_end(val['event_2_end'])
        datum.set_sentence_1(val['sentence_1'])
        datum.set_sentence_2(val['sentence_2'])
        datum.set_sentence_1_index(val['sentence_1_index'])
        datum.set_sentence_2_index(val['sentence_2_index'])
        datum.set_label(val['label'])
        return datum
    
    def to_dict(self):
        return {
            'text': self.text(),
            'event_1': self.event_1(),
            'event_2': self.event_2(),
            'event_1_start': self.event_1_start(),
            'event_1_end': self.event_1_end(),
            'event_2_start': self.event_2_start(),
            'event_2_end': self.event_2_end(),
            'sentence_1': self.sentence_1(),
            'sentence_2': self.sentence_2(),
            'sentence_1_index': self.sentence_1_index(),
            'sentence_2_index': self.sentence_2_index(),
            'label': self.label(),
        }
