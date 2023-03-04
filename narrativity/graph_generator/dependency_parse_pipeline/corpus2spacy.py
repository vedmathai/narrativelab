import coreferee, spacy

from narrativity.datamodel.featurized_document_model.featurized_document import FeaturizedDocument
from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence



class Corpus2spacy:
    def load(self):
        self._spacy = spacy.load('en_core_web_lg')
        self._spacy.add_pipe('coreferee')

    def convert(self, text):
        spacy_text = self._spacy(text)
        fdocument = FeaturizedDocument.from_spacy(spacy_text)
       
        return fdocument
