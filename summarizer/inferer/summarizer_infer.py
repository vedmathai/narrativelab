from memsum.src.summarizer import MemSum

from summarizer.common.config.config import Config


class SummarizerInfer:
    def __init__(self):
        self._config = Config.instance()

    def load(self, run_config={}):
        print(self._config.memsum_arxiv_model_path())
        self._memsum =  MemSum(self._config.memsum_arxiv_model_path(), 
                  self._config.memsum_vocab_path(), 
                  gpu = 0 ,  max_doc_len = 500)

    def classify(self, datum):
        output = self._model(datum)
        return output

    def infer(self, document):
        extracted_summary = self._memsum.extract([document], 
                                   p_stop_thres = 0.6, 
                                   max_extracted_sentences_per_document = 7,
                                   return_sentence_position=1
                                  )
        return extracted_summary

if __name__ == '__main__':
    summarizer = SummarizerInfer()
    summarizer.load()
    document = """
        Moctezuma II was the great-grandson of Moctezuma I through his daughter Atotoztli II and her husband Huehue Tezozómoc (not to be confused with the Tepanec leader). According to some sources, Tezozómoc was the son of emperor Itzcóatl, which would make Moctezuma his great-grandson, but other sources claim that Tezozómoc was Chimalpopoca's son, thus nephew of Itzcóatl, and a lord in Ecatepec. Moctezuma was also Nezahualcóyotl's grandson; he was a son of emperor Axayácatl and one of Nezahualcóyotl's daughters, Izelcoatzin or Xochicueyetl. Two of his uncles were Tízoc and Ahuizotl, the two previous emperors.

        As was customary among Mexica nobles, Moctezuma was educated in the Calmecac, the educational institution for the nobility. He would have been enrolled into the institution at a very early age, likely at the age of five years, as the sons of the kings were expected to receive their education at a much earlier age than the rest of the population. According to some sources, Moctezuma stood out in his childhood for his discipline during his education, finishing his works correctly and being devout to the Aztec religion.

        Moctezuma was an already famous warrior by the time he became the tlatoani of Mexico, holding the high rank of tlacatecuhtli (lord of men) and/or tlacochcalcatl (person from the house of darts) in the Mexica military, and thus his election was largely influenced by his military career and religious influence as a priest, as he was also the main priest of Huitzilopochtli's temple.
        Then-prince Moctezuma the Younger is arriving to the rescue of the merchants who were put under siege during the conquest of Ayotlan, according to the Florentine Codex. The merchants are seen talking to Moctezuma, informing him about the end of the war.

        One example of a celebrated campaign in which he participated before ascending to the throne was during the last stages of the conquest of Ayotlan, during Ahuizotl's reign in the late 15th century. During this campaign, which lasted 4 years, a group of Mexica pochteca merchants were put under siege by the enemy forces. This was important because the merchants were closely related to Ahuizotl and served as military commanders and soldiers themselves when needed. To rescue the merchants, Ahuizotl sent then-prince Moctezuma with many soldiers to fight against the enemies, though the fight didn't last long, as the people of Ayotlan surrendered to the Mexica shortly after he arrived.

        Approximately in the year 1490, Moctezuma obtained the rank of tequihua, which was reached by capturing at least 4 enemy commanders. 
    """
    summary = summarizer.infer(document.split('.'))
    print(summary)
