import json

from narrativity.graph_generator.dependency_parse_pipeline.parser import NarrativeGraphGenerator

text = "Alice wrote the paper with a typewriter and Bob submitted it"
text = "In September 1964, the world was worried about India's balance of payments situation. The World Bank and International Monetary Fund were having meetings in Tokyo, and IMF's Managing Director, Pierre-Paul Schweitzer, broached the topic with India's Finance Minister, T.T. Krishnamachari, known to most as TTK. The IMF was discreet, but there was something about the manner in which Schweitzer conveyed his concern that ticked off TTK. Participants at the meeting recall that TTK 'exploded' at Schweitzer."
text = "In September 1964, the world was worried about India's balance of payments situation."

def test_graph_generator_parser():
    ngg = NarrativeGraphGenerator()
    ngg.load()
    narrative_graph = ngg.generate(text)
    print(json.dumps(narrative_graph.to_dict(), indent=4))

    #ngg.generate('Although he was very busy with his work, Peter had had enough of it. He and his wife decided they needed a holiday. They travelled to Spain because they loved the country very much.')


if __name__ == '__main__':
    test_graph_generator_parser()
