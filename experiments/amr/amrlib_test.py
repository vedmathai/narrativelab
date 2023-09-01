import amrlib
import spacy
import penman
from penman import constant

gtos = amrlib.load_gtos_model()



amrlib.setup_spacy_extension()
nlp = spacy.load('en_core_web_lg')
doc = nlp('As he crossed toward the pharmacy at the corner he involuntarily turned his head because of a burst of light that had ricocheted from his temple, and saw, with that quick smile with which we greet a rainbow or a rose, a blindingly white parallelogram of sky being unloaded from the van—a dresser with mirrors across which, as across a cinema screen, passed a flawlessly clear reflection of boughs sliding and swaying not arboreally, but with a human vacillation, produced by the nature of those who were carrying this sky, these boughs, this gliding façade.')

# The following are roughly equivalent but demonstrate the different objects.
graphs = doc._.to_amr()
for graph in graphs:
    pass

sents, _ = gtos.generate(graphs)
for sent in sents:
    print(sent)

g = penman.decode(graph)
anon_map = {}
attributes = []
print(dir(g))
for instance in g.instances():
    print(instance)
#for edge in g.edges():
#    print(dir(edge.source))

