import React, { useState, useEffect } from 'react'
import { useNavigate } from "react-router-dom";


import './corpus-input.css'
import 'pages/pages.css'

import postSampleCorpusAPI from 'apis/corpus/postSampleCorpusAPI';
import getMostConnectedEntity from 'apis/nodes/getMostConnectedEntityAPI';
import TopBar from '../../common/top-bar/top-bar';


export default function CorpusInput(props) {
    var [corpus, setCorpus] = useState({"text": ""});
    var [nodes, setNodes] = useState([]);

    const navigate = useNavigate();

    const onChangeCorpusInputFn = (e) => {
        var c = {...corpus, "text": e.target.value}; 
        setCorpus(c);
    }

    const navigateToDiscover = async () => {
        navigate("/discover")
    }

    const onClickSubmitCorpus = async() => {
        await postSampleCorpusAPI(corpus);
        await navigateToDiscover();
    }

    useEffect(() => {
    }, []);

    return (
        <>
            <TopBar />
            <div className="page">
                <div className="page-content">
                    <div className="page-row">
                        <h2 className="page-heading">Enter the Sample Corpus:</h2>
                        <textarea
                            className="corpus-input-textarea"
                            onChange={(e) => onChangeCorpusInputFn(e)}
                        />
                    </div>
                    <div>
                        <button 
                            className="page-button corpus-input-submit-button"
                            onClick={() => onClickSubmitCorpus()}
                        >
                            Submit
                        </button>
                    </div>
                </div>
            </div>
        </>
    )
}
