class ModelInputDatum:
    """
        A catchall container for all the information about the datum.
        It can include raw information from the dataset, processed information,
        and even post processing information. Having all this information in
        one place makes it easy to use whichever information is important
        and even print it.
    """
    def __init__(self):
        self._sentence = None
        self._target = None

        self._is_trainable = False
        self._is_interested = False

    #  Getters and setters for all the fields.
    def sentence(self):
        return self._sentence
    
    def target(self):
        return self._target
    
    def is_trainable(self):
        return self._is_trainable
    
    def is_interested(self):
        return self._is_interested
    
    def set_sentence(self, sentence):
        self._sentence = sentence

    def set_target(self, target):
        self._target = target

    def set_is_trainable(self, is_trainable):
        self._is_trainable = is_trainable

    def set_is_interested(self, is_interested):
        self._is_interested = is_interested

    # A method that returns a dictionary of all the fields.
    def to_dict(self):
        return {
            'sentence': self._sentence,
            'target': self._target,
            'is_trainable': self._is_trainable,
            'is_interested': self._is_interested
        }
    
    # A method that takes a dictionary of the fields and sets them.
    @staticmethod
    def from_dict(datum_dict):
        datum = ModelInputDatum()
        datum.set_sentence(datum_dict['sentence'])
        datum.set_target(datum_dict['target'])
        datum.set_is_trainable(datum_dict['is_trainable'])
        datum.set_is_interested(datum_dict['is_interested'])
        return datum
