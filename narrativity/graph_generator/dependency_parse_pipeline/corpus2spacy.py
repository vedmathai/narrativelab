import coreferee, spacy

from narrativity.datamodel.featurized_document_model.featurized_document import FeaturizedDocument


class Corpus2spacy:
    def load(self):
        self._spacy = spacy.load('en_core_web_lg')
        self._spacy.add_pipe('coreferee')

    def convert(self, text):
        paragraphs = text.split('\n')
        spacy_paragraphs = []
        for paragraph in paragraphs:
            spacy_text = self._spacy(paragraph)
            spacy_paragraphs.append(spacy_text)
        fdocument = FeaturizedDocument.from_spacy(spacy_paragraphs)
        return fdocument
