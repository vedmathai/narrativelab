# Narrative Graph Datamodel
The graph's datamodel is split into nodes and relationship classes, and one class that points to the entire graph.

The node classes consist of:
1. Narrative node
2. Action Node
3. Entity Node
4. Absolute Temporal Node
5. State Node

The relationship classes consist of:
1. Object Relationship
2. Location Relationship
3. State Relationship
4. Temporal Relationship

These relationships are made into their own classes/objects so that they may hold their own attributes, for example, the location relationship may have a preposition attribute attached to it, like 'in' Oxford, 'near' Oxford.

The main holding class is the NarrativeGraph class. This maintains a list to all the nodes and relationships, which makes it useful to perform 'looping over all nodes' type operations. The NarrativeGraph class also maintains a mapping of id-to-node. This helps lookup the actual object given just the ID.

The next main class is that of the Narrative Node. Even though it is an instantiation of the Abstract Node just like the other types of nodes, it holds the most amount of information. Note that even though there are only a few classes, there may different types of instantiations of these classes, for example, direct objects, indirect object relationships are instantiations of the ObjectRelationship Class, and locations, actors, and objects are instantiations of the Object class. 

Every node and relationship is assigned a uuid (universally unique identifier), this makes it easy to serialize and deserialize. Since the relationships to objects do depend internal python pointers. However, wherever the objects point to another node, say a relationship, there are two parallel functions, one that returns the uuid of the relationship and the other that returns the object itself. For example, in the narrative node, there would be a `actor_ids()` and `actors()` methods. The first returns the list of ids that point to the actors of the given narrative, while the second returns the entity nodes of the actors. Internally, it uses the dictionaries in the NarrativeGraph to look up the objects given the IDs.

## The Classes
**Narrative Node:** A narrative can have either a set of actors, actions and objects, or be an action-state combination or have a list of sub-narratives. The narrative also is linked to its parent narratives if they exist. Narratives can also have a canonical name, location or temporal relationship. Below is a list of attributes for the narrative node:
1. `canonical_name`: The name given to a particular narrative, for example, 'The 2016 US elections.' 
2. `names`: Other names that the same narrative may take.
3. `actors`: The list of actors for the given narrative tuple (applicable at leaf level narratives).
4. `actions`: The list of actions performed in the given narrative (applicable at leaf level narratives).
5. `direct_object_relationships`: List of relationships to objects that the action was directly performed upon.
6. `indirect_object_relationships`: List of relationships to objects the the action was performed with or taking use of, for example, in the sentence, 'He hit the nail with the hammer,' the nail is the direct object, while the hammer is the indirect object.
7. `location_relationships`: List of relationships to locations where the action took place. The relationship captures the preposition to maintain relativity to the object.
8. `temporal_relationships`: List of relationships to the temporal location of the event.
9. `sub_narratives`: List of sub-narratives that together define the current narrative.
10. `parent_narratives`: List of narratives that is the parent of the given narrative. That is the mapping from sub-narrative to its parent.
11. `sources`: Mapping to list of sources in the original corpus from where this narrative was derived. 
12. `states`: The state descriptors for the narrative. Mostly for non-noun descriptors. "Biden is the President," here President would be an object.
13. `is_leaf`: Captures if the given narrative is a leaf or middle node.
14. `is_state`: Captures if the given narrative is capturing state or action.

**Entity Node:** An entity captures all entities that can be actors, objects or locations.

**Action Node:** Captures verbs or verb-like entities.

**State Node:** Captures states that the subject may be in. These may be adjectives in many cases, or capture the second part of 'is-a' relationships.

**Absolute Temporal Node:** Captures points in a time scale. It has a set of value and a unit attribute. For example, it can be February, 2023. Where year is 2023, and month is February. This needs further work, to differentiate it from durations or time periods.

**Location Relationship:** Captures the relationship between a narrative and a location entity, with an attribute for a preposition.

**Object Relationship:** Captures the relationship between a narrative and an object. Can be instantiated into a `direct_object_relationship` or an `indirect_object_relationship.`

**State Relationship:** Captures the relationship between a narrative and a state.

**Temporal Relationship:** Captures the relationship between a narrative and its absolute temporal markers. It should also capture the relative position of these temporal points in relation to the narrative.