import React, { useState, useEffect } from 'react'
import { useNavigate } from "react-router-dom";



import './node-context-row.css'

const key2headings = {
    'actor': 'Actors',
    'narrative': 'Narratives',
    'direct_object': 'Direct Objects',
    'indirect_object': 'Indirect Objects',
    'location': 'Locations',
    'subject': 'Subjects',
    'state_descriptor': 'State Descriptors',
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
        <div className='page-row node-context-row'>
            <div className="node-context-row-heading">
                {key2headings[props.k]}
            </div>
            <div className="node-context-page-cards">
                {pageCards}
            </div>
        </div>
    )
}
