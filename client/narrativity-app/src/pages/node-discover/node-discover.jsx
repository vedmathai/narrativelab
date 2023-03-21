import React, { useState, useEffect } from 'react'
import { useNavigate } from "react-router-dom";


import './node-discover.css'
import TopBar from 'common/top-bar/top-bar';
import 'pages/pages.css'
import NodeDiscoverSideBar from './components/node-discover-side-bar/node-discover-side-bar';
import searchGraphAPI from '../../apis/graph/graphSearchAPI';


export default function NodeDiscover(props) {
    var [searchResponse, setSearchResponse] = useState();
    var narratives = []
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

    if (searchResponse) {
        narratives = searchResponse.nodes.map((node, node_i) => {
            return (
                <div
                    className="page-card node-discover-page-card"
                    onClick={() => onClickContextRowCard(node.id)}
                >
                    {node.display_name}
                </div>
            )
        })
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
                    <div>
                        {narratives}
                    </div>                    
                </div>
            </div>
        </>
    )
}
