import React, { useState, useEffect } from 'react'
import { useNavigate } from "react-router-dom";



import './node-context-row.css'

const key2headings = {
    'actor': 'Actors',
    'narrative': 'Narratives',
    'direct_object': 'Direct Objects',
    'indirect_object': 'Indirect Objects',
    'location': 'Locations',
    'absolute_time': 'Absolute Temporal Relationships',
    'subject': 'Subjects',
    'state_descriptor': 'State Descriptors',
    'causal_in': 'Incoming Causal Relationship',
    'causal_out': 'Outgoing Causal Relationship',
    'contradictory_in': 'Incoming Contradictory Relationship',
    'contradictory_out': 'Outgoing Contradictory Relationship',
    'temporal_event_in': 'Incoming Temporal Event Relationship',
    'temporal_event_out': 'Outgoing Temporal Event Relationship',
    'anecdotal_in': 'Incoming Anecdotal Relationship',
    'anecdotal_out': 'Outgoing Anecdotal Relationship',
    'and_like': 'And Like Relationship',
    'descriptor': 'Describing Narrative',
    'described_entity': 'Described Entity',
    'trope': 'Tropes',
    'main_narrative_out': 'Outgoing Main Narrative Relationship',
    'main_narrative_in': 'Incoming Main Narrative Relationship',
}

export default function NodeContextRow(props) {
    const key2id = props.nodeContext.key2id;
    const id2node = props.nodeContext.id2node;

    const navigate = useNavigate();

    const onDoubleClickContextRowCard = (id) => {
        navigate("/node-context-explorer?node-id=" + id)
    }

    const pageCards = key2id[props.k].map((node_id, node_id_i) => {
        const node = id2node[node_id];
        var page_card_class_name = 'page-card node-context-page-card'
        if (node_id == props.focussedOtherNodeID) {
            page_card_class_name += ' node-context-page-card-focussed'
        }
        return (
            <div 
                className={page_card_class_name}
                onDoubleClick={() => onDoubleClickContextRowCard(node.id)}
                onClick={() => props.onClickContextRowCard(node_id)}
            >
                {node.display_name}
            </div>
        )
     })
    return (
        <div className='node-context-row'>
            <div className="page-card-row-heading">
                {key2headings[props.k]}
            </div>
            <div className="node-context-page-cards">
                {pageCards}
            </div>
        </div>
    )
}
