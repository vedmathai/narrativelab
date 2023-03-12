import React, { useState, useEffect } from 'react'
import { useSearchParams } from "react-router-dom";


import './node-context-explorer.css'
import TopBar from 'common/top-bar/top-bar';
import SideBar from 'common/side-bar/side-bar';
import 'pages/pages.css'
import getNodeContextAPI from 'apis/nodes/getNodeContextAPI';
import ContextNode from './components/context-node/context-node';
import NodeContextRow from './components/NodeContextRow/node-context-row';


export default function NodeContextExplorer(props) {
    var [nodeContext, setNodeContext] = useState();
    var contextNodeRow = ""
    var nodeContextRows = ""

    const [searchParams, setSearchParams] = useSearchParams();
    const node_id = searchParams.get("node-id");

    const getNodeContext = async (node_id) => {
        const context = await getNodeContextAPI(node_id);
        setNodeContext(context);
    }

    useEffect(() => {
        getNodeContext(node_id);
    }, [node_id]);

    console.log(typeof nodeContext != undefined, nodeContext);
    if (nodeContext) {
        contextNodeRow = <div className='page-row'>
            <ContextNode
                node={nodeContext.node}
            />
        </div>
        nodeContextRows = nodeContext.keys.map((k, k_i) => {
            return (
                <div className='page-row'>
                    {k}
                    <NodeContextRow
                        k={k}
                        nodeContext={nodeContext}
                    />
                </div>
            )
        })
    }

    return (
        <>
            <div className="page">
                <SideBar />
                <TopBar />
                <div className="page-content">
                   {contextNodeRow}
                   {nodeContextRows}

                </div>
            </div>
        </>
    )
}
