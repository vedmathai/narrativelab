import React, { useState, useEffect } from 'react'
import { useSearchParams } from "react-router-dom";


import './node-context-explorer.css'
import TopBar from 'common/top-bar/top-bar';
import 'pages/pages.css'
import getNodeContextAPI from 'apis/nodes/getNodeContextAPI';
import ContextNode from './components/context-node/context-node';
import NodeContextRow from './components/node-context-row/node-context-row';
import NodeContextSideBar from './components/node-context-side-bar/node-context-side-bar';


export default function NodeContextExplorer(props) {
    var [nodeContext, setNodeContext] = useState();
    var [focussedOtherNodeID, setFocussedOtherNodeID] = useState();
    var contextNodeRow = ""
    var nodeContextRows = ""

    const [searchParams, setSearchParams] = useSearchParams();
    const node_id = searchParams.get("node-id");

    const getNodeContext = async (node_id) => {
        const context = await getNodeContextAPI(node_id);
        setNodeContext(context);
    }

    const onClickContextRowCard = (id) => {
        setFocussedOtherNodeID(id);
    }

    useEffect(() => {
        getNodeContext(node_id);
    }, [node_id]);

    if (nodeContext) {
        contextNodeRow = <div className='page-row'>
            <div className="node-context-row-heading">Node under Observation</div>
            <ContextNode
                node={nodeContext.node}
            />
        </div>
        nodeContextRows = nodeContext.keys.map((k, k_i) => {
            return (
                <div className='page-row'>
                    <NodeContextRow
                        k={k}
                        nodeContext={nodeContext}
                        onClickContextRowCard={onClickContextRowCard}
                        focussedOtherNodeID={focussedOtherNodeID}
                    />
                </div>
            )
        })
    }

    return (
        <>
            <div className="page">
                <NodeContextSideBar 
                    focussedOtherNodeID = {focussedOtherNodeID}
                    nodeContext={nodeContext}
                />
                <TopBar />
                <div className="page-content">
                    {contextNodeRow}
                    {nodeContextRows}
                </div>
            </div>
        </>
    )
}
