import json
from jadelogs import JadeLogger


from narrativity.datamodel.extraction_paths.extraction_paths import ExtractionPaths

EXTRACTION_PATH_JSON = "narrativelab/narrativity/graph_generator/dependency_parse_pipeline/dependency2narrative/common/extraction_paths/extraction_paths.json"


class AbstractEvents2Relationships:
    def __init__(self):
        self._jade_logger = JadeLogger()

    def load(self):
        self._load_extraction_paths()

    def _load_extraction_paths(self):
        filepath = self._jade_logger.file_manager.code_filepath(EXTRACTION_PATH_JSON)
        with open(filepath) as f:
            extraction_paths_dict = json.load(f)
            extraction_paths = ExtractionPaths.from_dict(extraction_paths_dict)
        self._extraction_paths = extraction_paths

