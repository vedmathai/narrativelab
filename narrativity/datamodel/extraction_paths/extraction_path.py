from narrativity.datamodel.extraction_paths.extraction_path_element import ExtractionPathElement


class ExtractionPath:
    def __init__(self):
        self._element = None
        self._extraction_path = []

    def element(self):
        return self._element

    def extraction_path(self):
        return self._extraction_path

    def set_element(self, element):
        self._element = element

    def set_extraction_path(self, extraction_path):
        self._extraction_path = extraction_path

    def to_dict(self):
        return {
            "element": self.element(),
            "extraction_path": [i.to_dict() for i in self.extraction_path()]
        }

    @staticmethod
    def from_dict(val):
        extraction_path = ExtractionPath()
        extraction_path.set_element(val['element'])
        extraction_path_items = [ExtractionPathElement.from_dict(i) for i in val['extraction_path']]
        extraction_path.set_extraction_path(extraction_path_items)
        return extraction_path
