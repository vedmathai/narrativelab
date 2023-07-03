from narrativity.datamodel.narrative_graph.relationships.descriptor_relationship import DescriptorRelationship
from narrativity.graph_generator.dependency_parse_pipeline.dependency2narrative.events2relationships.abstract_events2relationships import AbstractEvents2Relationships


class Events2DescriptorRelationships(AbstractEvents2Relationships):  
    def __init__(self):
        super().__init__()

    def load(self):
        super().load()
 
    def extract(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        extraction_fns = [
            self._descriptor_relationship,
        ]
        for fn in extraction_fns:
            fn(narrative_1, narrative_2, phrase_connector, narrative_graph)
    
    def _descriptor_relationship(self, narrative_1, narrative_2, phrase_connector, narrative_graph):
        if phrase_connector.connector_type() in ['descriptor_relationship']:
            descriptor_relationship = DescriptorRelationship.create()
            descriptor_relationship.set_narrative_graph(narrative_graph)
            if len(narrative_1.actor_relationships()) > 0:
                for relationship in narrative_1.actor_relationships():
                    actor = relationship.actor()
                    if phrase_connector.connector_text().text() in actor.display_name():
                        descriptor_relationship.set_entity_node(actor)
                        actor.add_descriptor_relationship(descriptor_relationship)
                
            if len(narrative_1.subject_relationships()) > 0:
                for relationship in narrative_1.subject_relationships():
                    subject = relationship.subject()
                    if phrase_connector.connector_text().text() in subject.display_name():
                        descriptor_relationship.set_entity_node(subject)
                        subject.add_descriptor_relationship(descriptor_relationship)

            if len(narrative_1.direct_object_relationships()) > 0:
                for relationship in narrative_1.direct_object_relationships():
                    direct_object = relationship.object()
                    if phrase_connector.connector_text().text() in direct_object.display_name():
                        descriptor_relationship.set_entity_node(direct_object)
                        direct_object.add_descriptor_relationship(descriptor_relationship)

            if len(narrative_1.indirect_object_relationships()) > 0:
                for relationship in narrative_1.indirect_object_relationships():
                    indirect_object = relationship.object()
                    if phrase_connector.connector_text().text() in indirect_object.display_name():
                        descriptor_relationship.set_entity_node(indirect_object)
                        indirect_object.add_descriptor_relationship(descriptor_relationship)
            descriptor_relationship.set_narrative(narrative_2)
            narrative_2.add_descriptor_relationship(descriptor_relationship)
            narrative_graph.add_descriptor_relationship(descriptor_relationship)
