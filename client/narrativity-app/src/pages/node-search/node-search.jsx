import React, { useState, useEffect } from 'react'

import './node-search.css'
import TopBar from 'common/top-bar/top-bar';
import 'pages/pages.css'
import NodeSearchSideBar from './components/node-search-side-bar/node-search-side-bar';


export default function NodeSearch(props) {

    useEffect(() => {
    }, []);

    return (
        <>
            <div className="page">
                <NodeSearchSideBar />
                <TopBar />
            </div>
        </>
    )
}
