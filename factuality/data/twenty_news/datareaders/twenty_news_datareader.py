import bs4 as bs
import json
import os

from factuality.common.config import Config
from factuality.data.twenty_news.datamodels.twenty_news_datum import TwentyNewsDatum
from factuality.data.twenty_news.datamodels.twenty_news_data import TwentyNewsData

file_mapping = {
    'train': '20news-bydate/20news-bydate-train',
    'test': '20news-bydate/20news-bydate-test',
}

class TwentyNewsDataReader():
    def __init__(self):
        super().__init__()
        self._config = Config.instance()

    def twenty_news_data(self, data_type="train"):
        filename = self._config.twenty_news_data_location()
        folder_path = file_mapping[data_type]
        abs_folder_path = os.path.join(filename, folder_path)
        subject_folders = os.listdir(abs_folder_path)
        data = TwentyNewsData()
        for subject_folder in subject_folders:
            abs_subject_folderpath = os.path.join(abs_folder_path, subject_folder)
            articles = os.listdir(abs_subject_folderpath)
            for article in articles:
                abs_article_path = os.path.join(abs_subject_folderpath, article)
                with open(abs_article_path) as f:
                    text = f.read()
                    datum = TwentyNewsDatum()
                    datum.set_text(text)
                    datum.set_label(subject_folder)
                    data.add_datum(datum)
        return data


if __name__ == '__main__':
    twenty_news_data_reader = TwentyNewsDataReader()
    data = twenty_news_data_reader.twenty_news_data()
    print(data.data()[0].to_dict())
