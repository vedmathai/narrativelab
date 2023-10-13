import bs4 as bs
import json

from factuality.common.config import Config
from factuality.data.booksummaries.datamodels.book_summary_datum import BookSummaryDatum
from factuality.data.booksummaries.datamodels.book_summary_data import BookSummaryData


class BookSummariesDataReader():
    def __init__(self):
        super().__init__()
        self._config = Config.instance()

    def book_summaries_data(self):
        filename = self._config.book_summaries_data_location()
        data = BookSummaryData()
        with open(filename) as file:
            for line in file:
                split_line = line.split('\t')
                genre_dictionary = split_line[5]
                title = split_line[2]
                try:
                    genre_dictionary = json.loads(genre_dictionary)
                except:
                    continue
                genres = list(genre_dictionary.values())
                story = split_line[6]
                datum = BookSummaryDatum()
                datum.set_label(genres)
                datum.set_text(story)
                datum.set_title(title)
                data.add_datum(datum)
        return data


if __name__ == '__main__':
    bsmd = BookSummariesDataReader()
    data = bsmd.book_summaries_data()
    print(data.data()[0].to_dict())
