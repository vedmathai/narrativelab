import json
import os
from typing import List
from jadelogs import JadeLogger

from factuality.data.vitamin_c.readers.vitamin_c_model.vitamin_c_dataset import VitaminCDataset
from factuality.common.config import Config


class VitaminCDataReader():
    def __init__(self):
        super().__init__()
        self._config = Config.instance()
        self._filenames = self._config.vitamin_c_file_mapping()
        self._jadelogger = JadeLogger()


    def vitamin_c_dataset(self) -> VitaminCDataset:
        filename = self._filenames["train"]
        abs_filename = self._jadelogger.file_manager.data_filepath(filename)
        data_list = []
        with open(abs_filename) as f:
            for line in f:
                datum_dict = json.loads(line)
                data_list.append(datum_dict)
            dataset = VitaminCDataset.from_dict(data_list)
        return dataset
