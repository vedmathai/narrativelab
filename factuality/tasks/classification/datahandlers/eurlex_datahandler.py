from factuality.common.config import Config
from factuality.data.eurlex.datareaders.eurlex_datareader import EurlexDataReader

from factuality.tasks.classification.datamodels.classification_data import ClassificationData
from factuality.tasks.classification.datamodels.classification_datum import ClassificationDatum
from factuality.tasks.classification.datahandlers.abstract_datahandler import AbstractDataHandler


class EurlexInvertedDataHandler(AbstractDataHandler):
    def __init__(self):
        super().__init__()
        self._config = Config.instance()
        self._edr = EurlexDataReader()

    def data(self):
        classification_data = ClassificationData()
        for datum in self._edr.eurlex_data().data():
            classification_datum = ClassificationDatum()
            text_pieces = [
                datum.title(),
                datum.header(),
                datum.recitals(),
                datum.main_body(),
                datum.attachments(),
            ]
            text = " ".join(text_pieces)
            classification_datum.set_text(text)
            classification_datum.set_label(datum.label())
            classification_data.add_datum(classification_datum)
        return classification_data
