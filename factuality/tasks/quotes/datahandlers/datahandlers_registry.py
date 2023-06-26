from factuality.tasks.quotes.datahandlers.nela_datahandler import NelaDatahandler


class DatahandlersRegistry:
    _dict = {
        'nela': NelaDatahandler
    }

    def get_datahandler(self, name):
        handler = DatahandlersRegistry._dict[name]()
        handler.load()
        return handler
