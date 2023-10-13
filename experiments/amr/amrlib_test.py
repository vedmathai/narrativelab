import amrlib
import spacy
import penman

gtos = amrlib.load_gtos_model()



amrlib.setup_spacy_extension()
nlp = spacy.load('en_core_web_lg')
sentence = 'As he crossed toward the pharmacy at the corner he involuntarily turned his head because of a burst of light that had ricocheted from his temple, and saw, with that quick smile with which we greet a rainbow or a rose, a blindingly white parallelogram of sky being unloaded from the van—a dresser with mirrors across which, as across a cinema screen, passed a flawlessly clear reflection of boughs sliding and swaying not arboreally, but with a human vacillation, produced by the nature of those who were carrying this sky, these boughs, this gliding façade.'
sentence = """
Foxes are small to medium-sized, omnivorous mammals belonging to several genera of the family Canidae. They have a flattened skull, upright, triangular ears, a pointed, slightly upturned snout, and a long bushy tail ("brush").
        Twelve species belong to the monophyletic "true fox" group of genus Vulpes. Approximately another 25 current or extinct species are always or sometimes called foxes; these foxes are either part of the paraphyletic group of the South American foxes, or of the outlying group, which consists of the bat-eared fox, gray fox, and island fox.
        Foxes live on every continent except Antarctica. The most common and widespread species of fox is the red fox (Vulpes vulpes) with about 47 recognized subspecies. The global distribution of foxes, together with their widespread reputation for cunning, has contributed to their prominence in popular culture and folklore in many societies around the world. The hunting of foxes with packs of hounds, long an established pursuit in Europe, especially in the British Isles, was exported by European settlers to various parts of the New World. 
"""
doc = nlp(sentence)

# The following are roughly equivalent but demonstrate the different objects.
graphs = doc._.to_amr()
for graph in graphs:
    

    sents, _ = gtos.generate(graphs)
    for sent in sents:
        print(sent)

    g = penman.decode(graph)
    anon_map = {}
    attributes = []
    print(dir(g))
    for instance in g.instances():
        print(instance)
    for edge in g.edges():
        print(edge.source, edge.target, edge.role)

