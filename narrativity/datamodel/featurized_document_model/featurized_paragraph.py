from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence  # noqa
from narrativity.datamodel.featurized_document_model.utils import resolve_coreference_pointers


class FeaturizedParagraph:
    def __init__(self):
        self._sentences = []

    def add_sentence(self, sentence):
        self._sentences.append(sentence)

    def sentences(self):
        return self._sentences

    @staticmethod
    def from_spacy(para, fdoc):
        fpara = FeaturizedParagraph()
        for sentence in para.sents:
            fpara.add_sentence(FeaturizedSentence.from_spacy(sentence, fdoc))
        #fpara = resolve_coreference_pointers(fpara)
        return fpara
