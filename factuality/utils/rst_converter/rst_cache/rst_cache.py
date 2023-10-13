import json
import os

from factuality.common.config import Config
from factuality.utils.rst_converter.rst_datamodel import RSTNode


class RSTCache:
    def __init__(self):
        self._cache = {}
        self._data_name = None

    def add_rst(self, sentence, rst):
        self._cache[sentence] = rst

    def rst(self, sentence):
        return self._cache.get(sentence)
    
    def has_rst(self, sentence):
        return sentence in self._cache
    
    def set_data_name(self, data_name):
        self._data_name = data_name

    def data_name(self):
        return self._data_name
    
    def to_dict(self):
        d = {}
        for k, v in self._cache.items():
            d[k] = v.to_dict()
        return d
    
    @staticmethod
    def from_dict(d):
        cache = RSTCache()
        for k, v in d.items():
            cache.add_rst(k, RSTNode.from_dict(v))
        return cache
    
    def filename(self, data_name):
        config = Config.instance()
        rst_cache_location = config.rst_cache_location()
        return os.path.join(rst_cache_location, data_name + '.json')

    def save(self):
        filename = self.filename(self.data_name())
        with open(filename, 'wt') as f:
            json.dump(self.to_dict(), f)
            f.write(str(self.to_dict()))

    def load(self):
        filename = self.filename(self._data_name)
        if os.path.exists(filename):
            with open(filename, 'rt') as f:
                d = json.load(f)
                self._cache = RSTCache.from_dict(d)
