from factuality.common.config import Config
from factuality.data.booksummaries.datareaders.book_summaries_datareader import BookSummariesDataReader

from factuality.tasks.classification.datamodels.classification_data import ClassificationData
from factuality.tasks.classification.datamodels.classification_datum import ClassificationDatum
from factuality.tasks.classification.datahandlers.abstract_datahandler import AbstractDataHandler


class BookSummariesPairedDataHandler(AbstractDataHandler):
    def __init__(self):
        super().__init__()
        self._config = Config.instance()
        self._bsmd = BookSummariesDataReader()

    def data(self):
        classification_data = ClassificationData()
        for datum_i in range(0, len(self._bsmd.book_summaries_data().data()), 2):
            datum_1 = self._bsmd.book_summaries_data().data()[datum_i]
            datum_2 = self._bsmd.book_summaries_data().data()[datum_i + 1]
            classification_datum = ClassificationDatum()
            classification_datum.set_text(' '.join(datum_1.text(), datum_2.text()))
            classification_datum.set_label(datum_1.label() + datum_2.label())
            classification_data.add_datum(classification_datum)
        return classification_data
