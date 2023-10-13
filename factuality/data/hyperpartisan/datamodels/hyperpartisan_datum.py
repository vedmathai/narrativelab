class HyperpartisanDatum():
    def __init__(self):
        self._label = None
        self._text = None
        self._title = None

    def label(self):
        return self._label

    def set_label(self, label):
        self._label = label

    def text(self):
        return self._text

    def set_text(self, text):
        self._text = text

    def title(self):
        return self._title

    def set_title(self, title):
        self._title = title

    def to_dict(self):
        return {
            'label': self.label(),
            'text': self.text(),
            'title': self.title(),
        }

    @staticmethod
    def from_dict(data_dict):
        datum = HyperpartisanDatum()
        datum.set_label(data_dict['label'])
        datum.set_text(data_dict['text'])
        datum.set_title(data_dict['title'])
        return datum
