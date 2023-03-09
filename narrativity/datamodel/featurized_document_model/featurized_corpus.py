from narrativity.datamodel.featurized_document_model.featurized_document import FeaturizedDocument  # noqa


class FeaturizedCorpus:
    def __init__(self):
        self._documents = []

    def add_document(self, document):
        self._documents.append(document)

    def documents(self):
        return self._documents

    @staticmethod
    def from_spacy(corpus):
        fcorpus = FeaturizedCorpus()
        for document in corpus:
            fcorpus.add_document(FeaturizedDocument.from_spacy(document))
        return fcorpus
