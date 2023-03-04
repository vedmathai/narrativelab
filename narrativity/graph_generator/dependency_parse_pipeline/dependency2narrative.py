from narrativity.datamodel.featurized_document_model.featurized_sentence import FeaturizedSentence

path2key = {
    ('nsubj', 'ROOT'): 'actor',
    ('advcl', 'nsubj'): 'actor',
    ('advcl', 'acomp'): 'state',
    ('acomp', 'prep'): 'state',
    ('acomp', 'prep', 'pobj'): 'state',
    ('acomp', 'prep', 'pobj', 'poss'): 'state',
}


class Dependency2Narrative:

    def load(self):
        pass

    def convert(self, fdocument):
        for sentence in fdocument.sentences():
            for token_1 in sentence.tokens():
                for token_2 in sentence.tokens():
                    if token_1.i() != token_2.i():
                        path = FeaturizedSentence.dependency_path_between_tokens(token_1, token_2)
                        path_tuple = tuple([i.dep() for i in path])
                        if path_tuple in path2key:
                            print(token_1.text(), token_2.text(), path2key[path_tuple])


"""
doc = nlp('Although he was very busy with his work, Peter had had enough of it. He and his wife decided they needed a holiday. They travelled to Spain because they loved the country very much.')
doc._.coref_chains.print()
# Output:
#
# 0: he(1), his(6), Peter(9), He(16), his(18)
# 1: work(7), it(14)
# 2: [He(16); wife(19)], they(21), They(26), they(31)
# 3: Spain(29), country(34)
#
print(doc._.coref_chains.resolve(doc[31]))
# Output:
#
# [Peter, wife]
"""