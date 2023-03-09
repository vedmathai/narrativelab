import json

from narrativity.graph_generator.dependency_parse_pipeline.parser import NarrativeGraphGenerator

text = "Alice wrote the paper with a typewriter and Bob submitted it"
def test_graph_generator_parser():
    ngg = NarrativeGraphGenerator()
    ngg.load()
    narrative_graph = ngg.generate(text)
    print(json.dumps(narrative_graph.to_dict(), indent=4))

    #ngg.generate('Although he was very busy with his work, Peter had had enough of it. He and his wife decided they needed a holiday. They travelled to Spain because they loved the country very much.')


if __name__ == '__main__':
    test_graph_generator_parser()
