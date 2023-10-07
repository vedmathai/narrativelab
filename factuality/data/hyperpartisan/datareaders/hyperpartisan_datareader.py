import bs4 as bs
from jadelogs import JadeLogger
import json

from factuality.common.config import Config
from factuality.data.hyperpartisan.datamodels.hyperpartisan_datum import HyperpartisanDatum
from factuality.data.hyperpartisan.datamodels.hyperpartisan_data import HyperpartisanData


articles_dict = {
    "train": "hyperpartisan/articles-training-bypublisher-20181122.xml",
    "validaton": "hyperpartisan/articles-validation-bypublisher-20181122.xml",
}

labels_dict = {
    "train": "hyperpartisan/ground-truth-training-bypublisher-20181122.xml",
    "validation": "hyperpartisan/ground-truth-validation-bypublisher-20181122.xml",
}

class HyperpartisanDataReader():
    def __init__(self):
        super().__init__()
        self._config = Config.instance()
        self._jadelogger = JadeLogger()

    def hyperpartisan_data(self):
        filename = articles_dict["train"]
        abs_filename = self._jadelogger.file_manager.data_filepath(filename)
        data = HyperpartisanData()
        article2label = self.hyperpartisan_article2label()
        with open(abs_filename) as file:
            new_para = None
            for line in file:
                if line.startswith('<article'):
                    new_para = line
                elif new_para is not None:
                    new_para += line
                if new_para is not None:
                    soup = bs.BeautifulSoup(new_para, "lxml")
                    articles = soup.find_all('article')
                    for article in articles:
                        article_id = article.get('id')
                        title = article.get('title')
                        text = article.text
                        datum = HyperpartisanDatum()
                        datum.article_id = article_id
                        datum.set_label(article2label[article_id])
                        datum.set_text(text)
                        datum.set_title(title)
                        data.add_datum(datum)
                        print(json.dumps(datum.to_dict(), indent=4))
        return data
    
    def hyperpartisan_article2label(self, type="train"):
        filename = labels_dict[type]
        abs_filename = self._jadelogger.file_manager.data_filepath(filename)
        with open(abs_filename, 'r') as file:
            soup = bs.BeautifulSoup(file.read(), "lxml")
            article2label = {}
            articles = soup.find_all('article')
            for article in articles:
                article_id = article.get('id')
                article_hyperpartisan = article.get('hyperpartisan')
                article2label[article_id] = article_hyperpartisan
        return article2label


if __name__ == '__main__':
    hpdr = HyperpartisanDataReader()
    data = hpdr.hyperpartisan_data()
    print(data.data()[0].to_dict())
