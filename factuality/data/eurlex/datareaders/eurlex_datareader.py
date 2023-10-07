import bs4 as bs
import json
import os

from factuality.common.config import Config
from factuality.data.eurlex.datamodels.eurlex_datum import EurlexDatum
from factuality.data.eurlex.datamodels.eurlex_data import EurlexData

file_mapping = {
    'train': 'dataset/train',
    'test': 'dataset/test',
    'dev': 'dataset/dev',
}

class EurlexDataReader():
    def __init__(self):
        super().__init__()
        self._config = Config.instance()

    def eurlex_data(self, data_type="train"):
        filename = self._config.eurlex_data_location()
        folder_path = file_mapping[data_type]
        abs_folder_path = os.path.join(filename, folder_path)
        files = os.listdir(abs_folder_path)
        data = EurlexData()
        for filepath in files:
            abs_filepath = os.path.join(abs_folder_path, filepath)
            with open(abs_filepath) as f:
                json_data = json.loads(f.read()) 
                datum = EurlexDatum()
                datum.set_label(json_data['concepts'])
                datum.set_header(json_data['header'])
                datum.set_recitals(json_data['recitals'])
                datum.set_main_body(json_data['main_body'])
                datum.set_attachments(json_data['attachments'])
                datum.set_title(json_data['title'])
                data.add_datum(datum)
        return data


if __name__ == '__main__':
    eurlex_data_reader = EurlexDataReader()
    data = eurlex_data_reader.eurlex_data()
    print(data.data()[0].to_dict())
