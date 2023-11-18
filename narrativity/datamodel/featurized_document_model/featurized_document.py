from narrativity.datamodel.featurized_document_model.featurized_paragraph import FeaturizedParagraph  # noqa
from narrativity.datamodel.featurized_document_model.utils import resolve_coreference_pointers


class FeaturizedDocument:
    def __init__(self):
        self._paragraphs = []

    def add_paragraph(self, paragraph):
        self._paragraphs.append(paragraph)

    def paragraphs(self):
        return self._paragraphs

    @staticmethod
    def from_spacy(document):
        fdoc = FeaturizedDocument()
        for para in document:
            fdoc.add_paragraph(FeaturizedParagraph.from_spacy(para, fdoc))
        return fdoc
