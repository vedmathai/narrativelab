from tqdm import tqdm

from factuality.common.config import Config
from factuality.tasks.classification.datahandlers.datahandlers_registry import DatahandlersRegistry
from factuality.utils.rst_converter.rst_cache.rst_cache import RSTCache
from factuality.utils.rst_converter.sentence2rst import sentence2rst


class Data2RST:
    def __init__(self):
        self._config = Config.instance()
        self._datahandlers_registry = DatahandlersRegistry()
        self._data_names = [
            'booksummaries',
            'booksummaries_paired',
            'hyperpartisan',
            'twenty_news',
            'eurlex',
            'eurlex_inverted',
        ]

    def data2rst(self):
        for data_name in self._data_names:
            rst_cache = RSTCache()
            rst_cache.set_data_name(data_name)
            datahandler = self._datahandlers_registry.get_datahandler(data_name)
            data = datahandler.data().data()
            for d in tqdm(data):
                key = d.text().strip()
                if rst_cache.has_rst(key) is False:
                    rst = sentence2rst(key)
                    rst_cache.add_rst(key, rst)
                rst_cache.save()


if __name__ == '__main__':
    d2r = Data2RST()
    d2r.data2rst()