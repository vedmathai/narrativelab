import React, { useState, useEffect } from 'react'

import './corpus-input.css'

import postSampleCorpusAPI from 'apis/corpus/postSampleCorpusAPI';
import getNodesAPI from 'apis/nodes/getNodesAPI';


export default function CorpusInput(props) {
    var [corpus, setCorpus] = useState({"text": ""});
    var [nodes, setNodes] = useState([]);


    const onChangeCorpusInputFn = (e) => {
        var c = {...corpus, "text": e.target.value}; 
        setCorpus(c);
    }

    const onClickSubmitCorpus = async() => {
        await postSampleCorpusAPI(corpus);
        const nodes = await getNodesAPI();
        setNodes(nodes);
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
                    {JSON.stringify(nodes)}
                </div>
            </div>
        </>
    )
}
