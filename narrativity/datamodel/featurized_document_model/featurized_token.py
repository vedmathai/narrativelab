from collections import defaultdict


class FeaturizedToken:
    def __init__(self):
        self._text = None
        self._lemma = None
        self._tense = None
        self._aspect = None
        self._pos = None
        self._parent = None
        self._children = []
        self._i = None
        self._deps = defaultdict(list)
        self._dep = None
        self._tag = None
        self._vector = None
        self._i_in_sentence = None

    def text(self):
        return self._text

    def lemma(self):
        return self._lemma

    def aspect(self):
        return self._aspect

    def tense(self):
        return self._tense

    def tag(self):
        return self._tag

    def pos(self):
        return self._pos

    def vector(self):
        return self._vector
    
    def i_in_sentence(self):
        return self._i_in_sentence

    def children(self):
        return self._deps

    def all_children(self):
        children = []
        for deps_children in self._deps.values():
            children.extend(deps_children)
        return children

    def parent(self):
        return self._parent

    def dep(self):
        return self._dep

    def closest_parents(self, poses):
        token = self
        while token.pos() not in poses and token.dep() != 'ROOT':
            token = token.parent()
        if token.pos() in poses:
            return True, token
        if token.dep() == 'ROOT':
            return False, token

    def closest_children(self, poses):
        tokens = [self]
        token = self
        while len(tokens) > 0 and token.pos() not in poses:
            token = tokens.pop(0)
            tokens.extend(token.all_children())
        if token.pos() in poses:
            return True, token
        if len(tokens) == 0:
            return False, token

    def i(self):
        return self._i

    def set_text(self, text):
        self._text = text

    def set_lemma(self, lemma):
        self._lemma = lemma

    def set_aspect(self, aspect):
        self._aspect = aspect

    def set_tense(self, tense):
        self._tense = tense

    def set_pos(self, pos):
        self._pos = pos

    def set_dep(self, dep):
        self._dep = dep

    def set_tag(self, tag):
        self._tag = tag

    def set_vector(self, vector):
        self._vector = vector

    def set_children(self, children):
        self._children = children

    def set_parents(self, parents):
        self._parents = parents

    def set_i(self, i):
        self._i = i

    def set_i_in_sentence(self, i_in_sentence):
        self._i_in_sentence = i_in_sentence

    def add_child(self, child):
        self._deps[child.dep()].append(child)

    def set_parent(self, parent):
        self._parent = parent

    @staticmethod
    def from_spacy(token, sentence):
        ftoken = FeaturizedToken()
        ftoken._text = token.text
        ftoken._i = token.i
        ftoken._i_in_sentence = token.i - sentence.start
        ftoken._lemma = token.lemma_
        morph_dict = token.morph.to_dict()
        ftoken._tense = morph_dict.get('Tense')
        ftoken._aspect = morph_dict.get('Aspect')
        ftoken._tag = token.tag_
        ftoken._pos = token.pos_
        ftoken._dep = token.dep_
        ftoken._vector = token.vector
        return ftoken
