from bs4 import BeautifulSoup
import os
from collections import defaultdict


from coreference.common.config.config import Config
from coreference.datamodel.ecb.ecb_document import ECBDocument


class ECBReader:
    def __init__(self):
        self._config = Config.instance()
        self._ecb_path = self._config.ecb_path()

    def folders(self):
        folders = os.listdir(self._config.ecb_path())
        return folders

    def read_folder(self, folder):
        folder_path = os.path.join(self._ecb_path, folder)
        data = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                datum = f.read()
                self._bs = BeautifulSoup(datum, "xml")
                data.append(self._bs)
        return data
    
    def read_folders(self):
        folders = self.folders()
        for folder in folders:
            data = self.read_folder(folder)
            for datum in data:
                ecb_datum = ECBDocument.from_bs(datum)
                yield ecb_datum


if __name__ == '__main__':
    reader = ECBReader()
    reader.read_folders()
