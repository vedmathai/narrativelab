import React, { useState, useEffect } from 'react'
import { useNavigate } from "react-router-dom";


import './node-discover.css'
import TopBar from 'common/top-bar/top-bar';
import 'pages/pages.css'
import NodeDiscoverSideBar from './components/node-discover-side-bar/node-discover-side-bar';
import searchGraphAPI from '../../apis/graph/graphSearchAPI';


export default function NodeDiscover(props) {
    var [searchResponse, setSearchResponse] = useState();
    var narratives = [];
    var entities = [];
    var absolute_temporal = [];
    const navigate = useNavigate();

    const searchGraph = async () => {
        const response = await searchGraphAPI({});
        setSearchResponse(response);
    }
    useEffect(() => {
        searchGraph()
    }, []);

    const onClickContextRowCard = (id) => {
        navigate("/node-context-explorer?node-id=" + id)
    }

    const nodes2row = (nodes) => {
        const row_cards = nodes.map((node, node_i) => {
            return (
                <div
                    className={"page-card node-discover-page-card"}
                    onClick={() => onClickContextRowCard(node.id)}
                >
                    {node.display_name}
                    <div
                        className={(node.is_main? "node-discover-page-card-main-marker-on" : "node-discover-page-card-main-marker-off")}
                    >
                        <span class="main-narrative-marker"></span>
                    </div>
                </div>
            )
        })
        const row = <div className="page-card-row-content">
            {row_cards}
        </div>
        
        return row;
    }

    if (searchResponse) {
        narratives = nodes2row(searchResponse.narrative_nodes);
        entities = nodes2row(searchResponse.entity_nodes);
        absolute_temporal = nodes2row(searchResponse.absolute_temporal_nodes);
    }

    return (
        <>
            <div className="page">
                <NodeDiscoverSideBar />
                <TopBar />
                <div className="page-content">
                    <div className="page-heading">
                        Discover
                    </div>
                        <div className='page-container'>
                        <div className='page-card-row'>
                            <div className="page-card-row-heading">Narratives</div>
                            {narratives}
                        </div>
                        <div className='page-card-row'>
                            <div className="page-card-row-heading">Entities</div>
                            {entities}
                        </div>
                        <div className='page-card-row'>
                            <div className="page-card-row-heading">Absolute Temporal Nodes</div>
                            {absolute_temporal}
                        </div>
                    </div>                 
                </div>
            </div>
        </>
    )
}
