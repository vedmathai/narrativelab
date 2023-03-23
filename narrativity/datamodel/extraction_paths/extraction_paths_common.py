from narrativity.datamodel.extraction_paths.extraction_path_element import ExtractionPathElement


class ExtractionPathsCommon:
    def __init__(self):
        self._word_lists = {}

    def word_lists(self):
        return self._word_lists

    def keys2word_lists(self, keys):
        all_words = set()
        for key in keys:
            all_words |= self.key2word_list(key)
        return all_words

    def key2word_list(self, key):
        key = key.strip('$$')
        return set(self._word_lists[key])

    def set_word_lists(self, word_lists):
        self._word_lists = word_lists

    def to_dict(self):
        return {
            "word_lists": self.word_lists(),
        }

    @staticmethod
    def from_dict(val):
        extraction_path_common = ExtractionPathsCommon()
        extraction_path_common.set_word_lists(val['word_lists'])
        return extraction_path_common
