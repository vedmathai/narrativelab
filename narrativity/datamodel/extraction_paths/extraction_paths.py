from typing import List
from jadelogs import JadeLogger

from narrativity.datamodel.extraction_paths.extraction_path import ExtractionPath
from narrativity.datamodel.extraction_paths.extraction_paths_common import ExtractionPathsCommon


class ExtractionPaths:
    def __init__(self):
        self._extraction_paths = []
        self._common = None

    def element2extraction_paths(self, element):
        extraction_paths = []
        for ep in self._extraction_paths:
            if ep.element() == element:
                extraction_paths.append(ep)
        return extraction_paths

    def common(self):
        return self._common

    def extraction_paths(self) -> List[ExtractionPath]:
        return self._extraction_paths

    def set_extraction_paths(self, extraction_paths):
        self._extraction_paths = extraction_paths
    
    def set_common(self, common):
        self._common = common

    def to_dict(self):
        return {
            "extraction_paths": [i.to_dict() for i in self.extraction_paths()]
        }

    @staticmethod
    def from_dict(val):
        extraction_paths = ExtractionPaths()
        extraction_path_list = [ExtractionPath.from_dict(i) for i in val['extraction_paths']]
        extraction_paths.set_extraction_paths(extraction_path_list)
        extraction_paths.set_common(ExtractionPathsCommon.from_dict(val['common']))
        return extraction_paths
