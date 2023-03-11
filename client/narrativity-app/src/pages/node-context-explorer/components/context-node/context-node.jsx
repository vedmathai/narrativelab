import React, { useState, useEffect } from 'react'


import './context-node.css'


export default function ContextNode(props) {
    return (
        <>
            <div className="page-card">
                {props.node.display_name}
            </div>
        </>
    )
}
