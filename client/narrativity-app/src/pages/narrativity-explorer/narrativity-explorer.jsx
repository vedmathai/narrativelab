import React, { useState, useEffect } from 'react'

import './narrativity-explorer.css'
import TopBar from 'common/top-bar/top-bar';
import SideBar from 'common/side-bar/side-bar';
import CorpusInput from './components/corpus/corpus-input';
import 'pages/pages.css'


export default function NarrativityExplorer(props) {

    useEffect(() => {
    }, []);

    return (
        <>
            <div className="page">
                <SideBar />
                <TopBar />
                <CorpusInput />
            </div>
        </>
    )
}
