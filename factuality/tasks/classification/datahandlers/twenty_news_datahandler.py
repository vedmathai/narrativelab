from factuality.common.config import Config
from factuality.data.twenty_news.datareaders.twenty_news_datareader import TwentyNewsDataReader

from factuality.tasks.classification.datamodels.classification_data import ClassificationData
from factuality.tasks.classification.datamodels.classification_datum import ClassificationDatum
from factuality.tasks.classification.datahandlers.abstract_datahandler import AbstractDataHandler


class TwentyNewsDataHandler(AbstractDataHandler):
    def __init__(self):
        super().__init__()
        self._config = Config.instance()
        self._tndr = TwentyNewsDataReader()

    def data(self):
        classification_data = ClassificationData()
        for datum in self._tndr.twenty_news_data().data():
            classification_datum = ClassificationDatum()
            classification_datum.set_text(datum.text())
            classification_datum.set_label(datum.label())
            classification_data.add_datum(classification_datum)
        return classification_data
