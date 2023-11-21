from bs4 import BeautifulSoup
import os
from collections import defaultdict


from coreference.common.config.config import Config
from coreference.datamodel.ecb.ecb_document import ECBDocument


class ECBReader:
    def __init__(self):
        self._config = Config.instance()
        self._ecb_path = self._config.ecb_path()

    def folders(self):
        folders = os.listdir(self._config.ecb_path())
        return folders

    def read_folder(self, folder):
        folder_path = os.path.join(self._ecb_path, folder)
        data = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                datum = f.read()
                self._bs = BeautifulSoup(datum, "xml")
                data.append(self._bs)
        return data
    
    def read_folders(self):
        folders = self.folders()
        for folder in folders:
            data = self.read_folder(folder)
            for datum in data:
                ecb_datum = ECBDocument.from_bs(datum)
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
                    for s in sources:
                        tokens = m_id2token_id[s]
                        string = ''
                        for token in tokens:
                            string += token_id2token[token].text() + ' '
                        sentence_id = token_id2sentence_id[tokens[0]]
                        sentence = sentence_id2sentence[sentence_id]
                        if 'act' in coref.note().lower():
                            print(string, [i.text() for i in sentence])
                    print('------------------')
                print('------------------')







if __name__ == '__main__':
    reader = ECBReader()
    reader.read_folders()
