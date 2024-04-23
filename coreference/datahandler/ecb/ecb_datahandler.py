from collections import defaultdict
from transformers import RobertaTokenizerFast
import random


from coreference.datareader.ecb.ecb_reader import ECBReader
from coreference.datamodel.ecb.ecb_document import ECBDocument
from coreference.datamodel.llm_input.datum import CoreferenceLLMDatum


class ECBDatahandler:
    def __init__(self):
        self._reader = ECBReader()
        self._train_data = []
        self._test_data = []

    def train_data(self):
        return self._train_data
    
    def test_data(self):
        return self._test_data

    def load(self):
        data = []
        folders = self._reader.folders()
        for folder in folders:
            self.read_folder(folder, data)
        train_split = int(len(data) * 0.8)
        self._train_data = data[: train_split]
        self._test_data = data[train_split: ]
        return data
        
    def read_folder(self, folder, data):
        ecb_data = self._reader.read_folder(folder)
        for datum in ecb_data:
            self.read_datum(datum, data)


    def read_datum(self, datum, data):
        ecb_datum = ECBDocument.from_bs(datum)
        text = ' '.join([token.text() for token in ecb_datum.tokens()])
        sentence_id2sentence = defaultdict(list)
        token_id2sentence_id = {}
        m_id2token_id = {}
        token_id2token = {}
        for token in ecb_datum.tokens():
            token_id2token[token.tid()] = token
            sentence_id2sentence[token.sentence()].append(token)
            token_id2sentence_id[token.tid()] = token.sentence()
        for markable in ecb_datum.token_markables():
            m_id2token_id[markable.mid()] = markable.token_anchors()
        for coref in ecb_datum.cross_doc_corefs():
            sources = coref.sources()
            target = coref.target()

            for s1_i, s1 in enumerate(sources):
                for s2_i, s2 in enumerate(sources[s1_i + 1: ]):
                    tokens_1 = m_id2token_id[s1]
                    tokens_2 = m_id2token_id[s2]
                    string_1 = ''
                    for token in tokens_1:
                        string_1 += token_id2token[token].text() + ' '
                    string_2 = ''
                    for token in tokens_2:
                        string_2 += token_id2token[token].text() + ' '
                    sentence_1_id = token_id2sentence_id[tokens_1[0]]
                    sentence_1 = ' '.join([i.text() for i in sentence_id2sentence[sentence_1_id]])
                    sentence_2_id = token_id2sentence_id[tokens_2[0]]
                    sentence_2 = ' '.join([i.text() for i in sentence_id2sentence[sentence_2_id]])
                    llm_datum = CoreferenceLLMDatum()
                    llm_datum.set_text(text)
                    llm_datum.set_event_1(string_1)
                    llm_datum.set_event_1_start(int(tokens_1[0]))
                    llm_datum.set_event_1_end(int(tokens_1[-1]))
                    llm_datum.set_event_2(string_2)
                    llm_datum.set_event_2_start(int(tokens_2[0]))
                    llm_datum.set_event_2_end(int(tokens_2[-1]))
                    llm_datum.set_sentence_1(sentence_1)
                    llm_datum.set_sentence_2(sentence_2)
                    llm_datum.set_sentence_1_index(sentence_1_id)
                    llm_datum.set_sentence_2_index(sentence_2_id)
                    llm_datum.set_label(1)
                    data.append(llm_datum)
                    self._create_random(llm_datum, data)
        return data
    
    def _create_random(self, llm_datum, data):
        random_noise = random.randint(3, 10) * random.choice([-1, 1])
        llm_datum_2 = CoreferenceLLMDatum.from_dict(llm_datum.to_dict())
        llm_datum_2.set_event_1_start(llm_datum_2.event_1_start() + random_noise)
        llm_datum_2.set_event_1_end(llm_datum_2.event_1_end() + random_noise)
        llm_datum_2.set_label(0)
        data.append(llm_datum_2)


if __name__ == '__main__':
    datahandler = ECBDatahandler()
    train_data = datahandler.train_data()



