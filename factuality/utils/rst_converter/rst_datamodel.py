from nltk import Tree
import ast
import re


class RSTNode:
    def __init__(self):
        self._children = []
        self._label = None

    def add_child(self, child):
        self._children.append(child)

    def children(self):
        return self._children
    
    def set_label(self, label):
        self._label = label

    def label(self):
        return self._label
    
    @staticmethod
    def from_dict(d):
        node = RSTNode()
        node.set_label(d['label'])
        for child in d['children']:
            node.add_child(RSTNode.from_dict(child))
        return node
    
    def to_dict(self):
        d = {}
        d['label'] = self._label
        d['children'] = []
        for child in self._children:
            if isinstance(child, str):
                d['children'].append(child)
            else:
                d['children'].append(child.to_dict())
        return d

    @staticmethod
    def from_parse(parse):
        rst_node = RSTNode()
        string_children = []
        rst_node.set_label(parse.label())
        for child in parse:
            if isinstance(child, str):
                string_children.append(child)
            elif isinstance(child, Tree):
                strings = process_string_children(string_children)
                string_children = []
                for string in strings:
                    rst_node.add_child(string)
                rst_node.add_child(RSTNode.from_parse(child))
        strings = process_string_children(string_children)
        string_children = []
        for string in strings:
            rst_node.add_child(string)
        return rst_node
    

def process_string_children(string):
    og_string = string
    string = ' '.join(string)
    string = re.sub('ParseTree', '', string)
    if len(string.strip()) > 0:
        if string[-1] != ']':
            string += ']'
        if string[0] != '[':
            string = '[' + string
        try:
            string = ast.literal_eval(string)
        except:
            return [string]
    string = [i for i in string if len(i.strip()) > 0]
    return string