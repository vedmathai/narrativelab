import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import './index.css';
import reportWebVitals from './reportWebVitals';
import NodeContextExplorer from './pages/node-context-explorer/node-context-explorer';
import CorpusInput from './pages/corpus-input/corpus-input';
import NodeDiscover from './pages/node-discover/node-discover';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <Router>
    <Routes>
      <Route path="/node-context-explorer" element={<NodeContextExplorer />} />
      <Route path="/discover" element={<NodeDiscover />} />
      <Route path="/" element={<CorpusInput />} />

    </Routes>
</Router>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
