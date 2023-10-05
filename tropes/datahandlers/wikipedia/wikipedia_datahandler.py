import bz2
import csv
import os

from tropes.common.config.config import Config


class WikipediaDatahandler():
    # write init function
    def __init__(self):
        self._config = Config.instance()
        self._wikipedia_files = []
        
    def load(self):
        self._wikipedia_files = os.listdir(self._config.wikipedia_data_location())

    def data(self):
        for file in self._wikipedia_files:
            print(file)
            filepath = os.path.join(self._config.wikipedia_data_location(), file)
            with bz2.BZ2File(filepath, "r") as f:
                for line in f:
                    yield line.decode('utf-8')

