from collections import defaultdict
import json
import numpy as np
import csv
from collections import defaultdict

from factuality.data.vitamin_c.readers.vitamin_c_datareader import VitaminCDataReader


class VitaminCStats():
    def __init__(self):
        self._vitamin_c_datareader = VitaminCDataReader()

    def all_sentences(self):
        dataset = self._vitamin_c_datareader.vitamin_c_dataset()
        cases = set()
        claims = set()
        evidence = set()
        dd = defaultdict(int)
        sentences = []


        for datum in dataset.data():
            cases.add(datum.case_id())
            #claims.add(datum.claim())
            evidence.add(datum.evidence())
            k = len(datum.claim())
            #dd[int(len(datum.claim().split()) / 10)] += 1
            dd[int(len(datum.evidence().split()) / 10)] += 1
            if int(len(datum.evidence().split()) / 10) == 2:
                sentences.append(datum.evidence())
        d = sorted(dd.items(), key=lambda x: x[1])
        with open('sentences.csv', 'wt') as f:
            writer = csv.writer(f, delimiter='\t')
            for i in sentences:
                writer.writerow([i])
        print(len(dataset.data()), len(cases), len(claims), len(evidence))