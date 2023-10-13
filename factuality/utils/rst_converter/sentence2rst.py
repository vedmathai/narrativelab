import requests
from nltk import Tree
import pprint
import json
import re

from factuality.utils.rst_converter.rst_datamodel import RSTNode


def sentence2rst(sentence):
    url = 'http://localhost:8000/parse'
    files = {
        'input': (None, sentence),
    }

    response = requests.post(url, files=files)
    if response.status_code == 500:
        rst_node = RSTNode()
        rst_node.add_child(sentence)
    else:
        parse = '({})'.format(response.text)
        tree = Tree.fromstring(parse)
        rst_node = RSTNode.from_parse(tree)
    return rst_node


if __name__ == '__main__':
    sentence = """There is a persistent urban legend that you can walk between Oxford and Cambridge without ever leaving college land.

                The ancient colleges of Oxford University are certainly wealthy institutions: the combined funds of all the colleges in 2015 were revealed to be some £4.1 billion, with £1.3 billion invested in property. It’s clear that beyond the dreaming spires of the colleges themselves, with their honey-coloured quads dating back to medieval times and cobbled streets strewn with post-examination revelries, stretch property empires worth many millions of pounds.

                Wealth brings with it great power, and in the words of Spider-Man’s mentor, ‘With great power comes great responsibility’. I had the privilege of studying at Oxford, and certainly benefited from its wealth and the education it helped paid for. Clearly, Oxford has wider social responsibilities – from ensuring anyone, from any background, can gain an education there; to investing its money ethically (such as by divesting from all fossil fuels, which the University has so far failed to do). I believe the Oxford Colleges should also be open about what they own. So, I asked them to tell me.

                I began with a Freedom of Information request to the central University of Oxford, asking for a GIS map of all the land owned by the University and its colleges. But I had forgotten how fiercely independent each college remains. I was told: “The University’s Estates Department does not hold a GIS map of all University land or college land. Please note that colleges are separate public authorities under the Freedom of Information Act”. "
    """
    sentence = re.sub('\(', '', sentence)
    sentence = re.sub('\)', '', sentence)

    rst = sentence2rst(sentence)
    print(json.dumps(rst.to_dict(), indent=4))
