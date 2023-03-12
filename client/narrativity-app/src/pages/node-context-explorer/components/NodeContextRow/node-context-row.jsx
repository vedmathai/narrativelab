import React, { useState, useEffect } from 'react'
import { useNavigate } from "react-router-dom";



import './node-context-row.css'

const key2headings = {
    'actors': "Actors",
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
        return (
            <div 
                className='page-card'
                onDoubleClick={() => onDoubleClickContextRowCard(node.id)}
            >
                {node.display_name}
            </div>
        )
     })
    return (
        <div className='page-row node-context-rows'>
            {pageCards}
        </div>
    )
}
