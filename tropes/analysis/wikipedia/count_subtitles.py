from collections import defaultdict
import re

from tropes.datahandlers.wikipedia.wikipedia_datahandler import WikipediaDatahandler

class WikipediaAnalysis:
    def __init__(self):
        self._wikipedia_data_handler = WikipediaDatahandler()
        self._subtitles_counts = defaultdict(int)

    def load(self):
        self._wikipedia_data_handler.load()

    def analyse(self):
        for datum in self._wikipedia_data_handler.data():
            for line in datum.split('\n'):
                if line.startswith('==') and not line.startswith('==='):
                    word = re.sub(r'=*', '', line)
                    word = word.strip()
                    self._subtitles_counts[word] += 1

        print(sorted(self._subtitles_counts.items(), key=lambda x: x[1]))


if __name__ == '__main__':
    wa = WikipediaAnalysis()
    wa.load()
    wa.analyse()
