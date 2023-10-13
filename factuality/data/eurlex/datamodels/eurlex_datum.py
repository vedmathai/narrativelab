class EurlexDatum():
    def __init__(self):
        self._label = None
        self._title = None
        self._label = None
        self._header = None
        self._recitals = None
        self._main_body = None
        self._attachments = None

    def label(self):
        return self._label

    def set_label(self, label):
        self._label = label

    def header(self):
        return self._header

    def set_header(self, header):
        self._header = header

    def recitals(self):
        return self._recitals

    def set_recitals(self, recitals):
        self._recitals = recitals

    def main_body(self):
        return self._main_body

    def set_main_body(self, main_body):
        self._main_body = main_body

    def attachments(self):
        return self._attachments

    def set_attachments(self, attachments):
        self._attachments = attachments

    def title(self):
        return self._title

    def set_title(self, title):
        self._title = title

    def to_dict(self):
        return {
            'label': self.label(),
            'header': self.header(),
            'recitals': self.recitals(),
            'main_body': self.main_body(),
            'attachments': self.attachments(),
            'title': self.title()
        }

    @staticmethod
    def from_dict(data_dict):
        datum = EurlexDatum()
        datum.set_label(data_dict['label'])
        datum.set_header(data_dict['header'])
        datum.set_recitals(data_dict['recitals'])
        datum.set_main_body(data_dict['main_body'])
        datum.set_attachments(data_dict['attachments'])
        datum.set_title(data_dict['title'])
        return datum
