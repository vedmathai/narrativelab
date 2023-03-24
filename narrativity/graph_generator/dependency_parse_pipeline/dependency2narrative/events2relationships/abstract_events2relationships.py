import json

from narrativity.datamodel.extraction_paths.extraction_paths import ExtractionPaths

EXTRACTION_PATH_JSON = "narrativity/graph_generator/dependency_parse_pipeline/dependency2narrative/common/extraction_paths/extraction_paths.json"


class AbstractEvents2Relationships:
    def load(self):
        self._load_extraction_paths()

    def _load_extraction_paths(self):
        with open(EXTRACTION_PATH_JSON) as f:
            extraction_paths_dict = json.load(f)
            extraction_paths = ExtractionPaths.from_dict(extraction_paths_dict)
        self._extraction_paths = extraction_paths

