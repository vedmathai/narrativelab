import json

from narrativity.graph_generator.dependency_parse_pipeline.parser import NarrativeGraphGenerator

text = "Alice wrote the paper with a typewriter and Bob submitted it."


def test_graph_generator_parser():
    ngg = NarrativeGraphGenerator()
    ngg.load()
    narrative_graph = ngg.generate(text)
    print(json.dumps(narrative_graph.to_dict(), indent=4))


if __name__ == '__main__':
    test_graph_generator_parser()
