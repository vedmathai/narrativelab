import json
import os
from jadelogs import JadeLogger

from narrativity.datamodel.extraction_paths.extraction_paths import ExtractionPaths


EXTRACTION_PATH_JSON = "narrativelab/narrativity/graph_generator/dependency_parse_pipeline/dependency2narrative/common/extraction_paths/extraction_paths.json"

class ExtractionPathMatcher:
    def __init__(self):
        self._environment = os.environ.get('ENVIRONMENT')
        self._jade_logger = JadeLogger()
        self._extraction_paths = self.load_extraction_paths()
        self._extraction_paths_common = self._extraction_paths.common()
        
    def load_extraction_paths(self):
        filepath = EXTRACTION_PATH_JSON
        if self._environment == 'JADE':
            filepath = self._jade_logger.file_manager.code_filepath(filepath)
        with open(filepath) as f:
            extraction_paths_dict = json.load(f)
            extraction_paths = ExtractionPaths.from_dict(extraction_paths_dict)
        return extraction_paths

    def match(self, path, element_name):
        extraction_paths = self._extraction_paths.element2extraction_paths(element_name)
        extraction_paths = [i for i in extraction_paths if len(i.extraction_path()) == len(path)]  # Match only paths that are exactly equal in size
        for elementi, element in enumerate(path):
            filtered_extraction_paths = []
            for extraction_path in extraction_paths:
                if self.check_path_element(elementi, element, extraction_path) is True:
                    filtered_extraction_paths.append(extraction_path)
            extraction_paths = filtered_extraction_paths
        if len(extraction_paths) > 0:
            return True
        return False

    def check_path_element(self, elementi, element, extraction_path):
        path_element = extraction_path.extraction_path()[elementi]
        dep_check = element.dep() in path_element.deps()
        pos_check = True
        if len(path_element.pos()) > 0:
            pos_check = element.pos() in path_element.pos()
        tokens_check = True
        token_whitelists = path_element.tokens()
        token_whitelist = self._extraction_paths_common.keys2word_lists(token_whitelists)
        if len(token_whitelist) > 0 and token_whitelists:
            tokens_check = element.text().lower() in token_whitelist
        token_blacklists = path_element.tokens_blacklist()
        blacklist_tokens = self._extraction_paths_common.keys2word_lists(token_blacklists)
        blacklist_tokens_check = True
        if len(blacklist_tokens) > 0 and element.text().lower() in blacklist_tokens:
            blacklist_tokens_check = False
        entity_type_check = self._check_entity_type(element, path_element)
        checks = [dep_check, pos_check, tokens_check, blacklist_tokens_check, entity_type_check]
        check = all(i for i in checks)
        return check

    def _check_entity_type(self, element, path_element):
        entity_type_check = True
        entity_types_whitelists = path_element.entity_types()
        if len(entity_types_whitelists) > 0 and entity_types_whitelists:
            entity_type_check = element.entity_type() in entity_types_whitelists
        blacklist_entity_types = path_element.entity_types_blacklist()
        blacklist_entity_types_check = True
        if len(blacklist_entity_types) > 0 and element.entity_type() in blacklist_entity_types:
            blacklist_entity_types_check = False
        return entity_type_check and blacklist_entity_types_check