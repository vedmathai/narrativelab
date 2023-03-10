import React, { useState, useEffect } from 'react'
import { useNavigate } from "react-router-dom";


import './corpus-input.css'
import 'pages/pages.css'

import postSampleCorpusAPI from 'apis/corpus/postSampleCorpusAPI';
import getMostConnectedEntity from 'apis/nodes/getMostConnectedEntityAPI';


export default function CorpusInput(props) {
    var [corpus, setCorpus] = useState({"text": ""});
    var [nodes, setNodes] = useState([]);

    const navigate = useNavigate();

    const onChangeCorpusInputFn = (e) => {
        var c = {...corpus, "text": e.target.value}; 
        setCorpus(c);
    }

    const navigateToNodeContext = async (id) => {
        navigate("/node-context-explorer?node-id=" +id)
    }

    const onClickSubmitCorpus = async() => {
        await postSampleCorpusAPI(corpus);
        const entity = await getMostConnectedEntity();
        await navigateToNodeContext(entity.id);
    }

    useEffect(() => {
    }, []);

    return (
        <>
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
                            className="corpus-input-button"
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
