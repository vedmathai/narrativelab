#write the import for TVTropesDataHandler
from tropes.datahandlers.tv_tropes.tv_tropes_datahandler import TVTropesDataHandler


class DataHandlerRegistry():
    _registry = {
        'tvtropes': TVTropesDataHandler,
    }

    _instance = None

    @staticmethod
    def instance():
        if DataHandlerRegistry._instance is None:
            DataHandlerRegistry._instance = DataHandlerRegistry()
        return DataHandlerRegistry._instance

    def __init__(self):
        super().__init__()

    def get_data_handler(self, data_handler_name: str):
        """Get data handler by name."""
        return self._registry[data_handler_name]()
