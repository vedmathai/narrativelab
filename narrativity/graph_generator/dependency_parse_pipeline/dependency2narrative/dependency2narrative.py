from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.sentence2phrases import Sentence2Phrases
from narrativity.datamodel.narrative_graph.narrative_graph import NarrativeGraph
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.verb2event.verb2event import Verb2Event
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.aux2state.aux2state import Aux2State
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.events2relationships.events2relationships import Events2Relationships
from narrativity.datamodel.narrative_graph.relationships.cooccurrence_relationship import CooccurrenceRelationship


class Dependency2Narrative:
    def __init__(self):
        self._sentence2phrases = Sentence2Phrases()
        self._verb2event = Verb2Event()
        self._aux2state = Aux2State()
        self._events2relationships = Events2Relationships()

    def load(self):
        self._sentence2phrases.load()
        self._verb2event.load()
        self._aux2state.load()
        self._events2relationships.load()

    def convert(self, fdocument):
        self._narrative_graph = NarrativeGraph()
        phrase_connectors = []
        annecdotal_connectors = []
        tokeni2event = {}
        all_narratives = []
        for sentence in fdocument.sentences():
            if sentence.root() is not None:
                phrase_connectors = self._sentence2phrases.split(sentence.root(), phrase_connectors)
        for phrase_connector in phrase_connectors:
            verb_1 = phrase_connector.verb_1()
            verb_2 = phrase_connector.verb_2()
            narrative_node_1 = tokeni2event.get(verb_1.i())
            if narrative_node_1 is None:
                narrative_node_1 = self._clause_root2event(verb_1)
                tokeni2event[verb_1.i()] = narrative_node_1
                narrative_node_1._token = verb_1
                all_narratives.append(narrative_node_1)
            if phrase_connector.connector_type() != 'single':
                narrative_node_2 = tokeni2event.get(verb_2.i())
                if narrative_node_2 is None:
                    narrative_node_2 = self._clause_root2event(verb_2)
                    tokeni2event[verb_2.i()] = narrative_node_2
                    narrative_node_2._token = verb_2
                    all_narratives.append(narrative_node_2)
                self._create_event_relationships(narrative_node_1, narrative_node_2, phrase_connector)
            if phrase_connector.connector_type() == 'anecdotal_relationship':
                annecdotal_connectors.append((narrative_node_1, narrative_node_2, phrase_connector))
        self._create_anecdotal_event_relationships(annecdotal_connectors)
        self._create_cooccurrence_relationships(all_narratives)
        return self._narrative_graph

    def _clause_root2event(self, clause_root):
        if clause_root is None:
            return
        if clause_root.pos() == 'VERB':
            narrative_node = self._verb2event.convert(clause_root, self._narrative_graph)
        if clause_root.pos() == 'AUX':
            narrative_node = self._aux2state.convert(clause_root, self._narrative_graph)
        return narrative_node

    def _create_event_relationships(self, narrative_node_1, narrative_node_2, phrase_connector):
        self._events2relationships.extract(narrative_node_1, narrative_node_2, phrase_connector, self._narrative_graph)

    def _create_anecdotal_event_relationships(self, annecdotal_connectors):
        # This has to be processed after every other relationship because anecdotal relationships point can point
        # to relationships instead of just narrative node.
        for annecdotal_connector in annecdotal_connectors:
            narrative_node_1, narrative_node_2, phrase_connector = annecdotal_connector
            self._events2relationships.extract_annecdotal_relationships(narrative_node_1, narrative_node_2, phrase_connector, self._narrative_graph)

    def _create_cooccurrence_relationships(self, all_narratives):
        seen = set()
        all_narratives_set = []
        for narrative in all_narratives:
            if narrative.id() not in all_narratives_set:
                all_narratives_set.append(narrative)
        for narrative_1 in all_narratives:
            for narrative_2 in all_narratives:
                if narrative_1.id() != narrative_2.id():
                    cooccurrence_relationship = CooccurrenceRelationship.create()
                    narrative_1.add_cooccurrence_relationship(cooccurrence_relationship)
                    narrative_2.add_cooccurrence_relationship(cooccurrence_relationship)
                    cooccurrence_relationship.set_narrative_graph(self._narrative_graph)
                    cooccurrence_relationship.set_narrative_1(narrative_1)
                    cooccurrence_relationship.set_narrative_2(narrative_2)
                    self._narrative_graph.add_cooccurrence_relationship(cooccurrence_relationship)
