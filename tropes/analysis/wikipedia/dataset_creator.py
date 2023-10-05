from collections import defaultdict
import re
import jadelogs
import wikitextparser
import csv


from tropes.datahandlers.wikipedia.wikipedia_datahandler import WikipediaDatahandler

class WikipediaAnalysis:
    def __init__(self):
        self._wikipedia_data_handler = WikipediaDatahandler()
        self._subtitles_counts = defaultdict(int)
        self._jade_logger = jadelogs.JadeLogger()

    def load(self):
        self._wikipedia_data_handler.load()

    def get_list_of_subheadings(self):
        subheadings = []
        location = 'narrativelab/tropes/analysis/wikipedia/list_of_picked_subheadings.py'
        location = self._jade_logger.file_manager.code_filepath(location)
        with open(location) as f:
            for line in f:
                subheadings.append(line.strip())
        return subheadings

    def create_dataset(self):
        dataset = defaultdict(list)
        subheadings = self.get_list_of_subheadings()
        for subheading in subheadings:
            dataset[subheading] = []
        interested = False
        para = ""

        for datum in self._wikipedia_data_handler.data():
            heading_flag = datum.startswith('==') and not datum.startswith('===')
            if heading_flag is True:
                word = re.sub(r'=*', '', datum)
                word = word.strip()
                if word in subheadings:
                    interested = True
                else:
                    interested = False
                if para != "":
                    if interested is True:
                        dataset[key].extend(self._clean_sentence(para))
                    para = ""
                key = word
            if interested is True and heading_flag is False:
                para += ' ' + datum
            if all(len(dataset[key]) > 1000  for key in dataset.keys()):
                for key in dataset:
                    print(key)
                    print('--' * 4),
                    print(dataset[key])
                break
        self.dict2csv(dataset)

    def _clean_sentence(self, sentence):
        if sentence.strip() == '':
            return []
        sections = wikitextparser.parse(sentence).sections
        text = ""
        for section in sections:
            try:
                text += section.plain_text()
            except (IndexError, AttributeError):
                continue
        text = text.split('. ')
        text = [i + '. ' for i in text if len(i) > 0]
        return text
    
    def dict2csv(self, dict):
        location = 'narrativelab/tropes/analysis/wikipedia/output.csv'
        location = self._jade_logger.file_manager.code_filepath(location)
        writer = csv.writer(open(location, 'wt'))
        for key, value in dict.items():
            for line in value:
                line = re.sub(r'\n', '', line)
                if 'ref' in line or line.strip() == '' or len(line) < 10:
                    continue
                writer.writerow([key, line])


if __name__ == '__main__':
    wa = WikipediaAnalysis()
    wa.load()
    wa.create_dataset()
