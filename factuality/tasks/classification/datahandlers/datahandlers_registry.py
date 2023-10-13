from factuality.tasks.classification.datahandlers.booksummaries_datahandler import BookSummariesDataHandler
from factuality.tasks.classification.datahandlers.booksummaries_paired_datahandler import BookSummariesPairedDataHandler
from factuality.tasks.classification.datahandlers.hyperpartisan_datahandler import HyperPartisanDataHandler
from factuality.tasks.classification.datahandlers.twenty_news_datahandler import TwentyNewsDataHandler
from factuality.tasks.classification.datahandlers.booksummaries_paired_datahandler import BookSummariesPairedDataHandler
from factuality.tasks.classification.datahandlers.eurlex_datahandler import EurlexDataReader
from factuality.tasks.classification.datahandlers.eurlex_inverted_datahandler import EurlexInvertedDataHandler


class DatahandlersRegistry:
    _dict = {
        'booksummaries': BookSummariesDataHandler,
        'booksummaries_paired': BookSummariesPairedDataHandler,
        'hyperpartisan': HyperPartisanDataHandler,
        'twenty_news':  TwentyNewsDataHandler,
        'eurlex': EurlexDataReader,
        'eurlex_inverted': EurlexInvertedDataHandler,
    }

    def get_datahandler(self, name):
        handler = DatahandlersRegistry._dict[name]()
        handler.load()
        return handler
