from factuality.common.config import Config
from factuality.data.booksummaries.datareaders.book_summaries_datareader import BookSummariesDataReader
from factuality.tasks.classification.datamodels.classification_data import ClassificationData
from factuality.tasks.classification.datamodels.classification_datum import ClassificationDatum
from factuality.tasks.classification.datahandlers.abstract_datahandler import AbstractDataHandler


class BookSummariesDataHandler(AbstractDataHandler):
    def __init__(self):
        super().__init__()
        self._config = Config.instance()
        self._bsmd = BookSummariesDataReader()

    def data(self):
        classification_data = ClassificationData()
        for datum in self._bsmd.book_summaries_data().data():
            classification_datum = ClassificationDatum()
            classification_datum.set_text(datum.text())
            classification_datum.set_label(datum.label())
            classification_data.add_datum(classification_datum)
        return classification_data
