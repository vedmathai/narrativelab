import { useNavigate } from "react-router-dom";

import jade_logo from 'common/assets/oxford-logo.png';


import './node-search-side-bar.css'


function SideBarRow(props) {
    const navigate = useNavigate();
    const onClickSideBarRow = () => {
        navigate(props.url);
    }
    return (
        <div className="side-bar-row"
            onClick={() => onClickSideBarRow()}
        >
            {props.label}
        </div>
    )
}

export default function NodeSearchSideBar(props) {
    const side_bar_rows_dict = [
        {
            'label': 'Corpus',
            'url': '/corpus-input',
        }
    ]
    const side_bar_rows = side_bar_rows_dict.map((row_dict) => {
        return <SideBarRow 
            label={row_dict.label}
            url={row_dict.url}
        />
    })
        
    return (
        <div className="side-bar">
            <div className="side-bar-logo">
                <img className="side-bar-logo-image" src={jade_logo} />
            </div>
            <div className='sidebar-rows'>
                {side_bar_rows}
            </div>
        </div>
    )
}
