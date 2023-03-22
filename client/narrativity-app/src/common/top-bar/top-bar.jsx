import { useNavigate } from "react-router-dom";

import './top-bar.css'


export default function TopBar(props) {
    const navigate = useNavigate();
    
    const onClickTopBarNavLink = (id) => {
        const link_map = {
            'home': '/',
            'discover': '/discover',
            'content-explorer': '/content-explorer'            
        }
        console.log(link_map[id])
        navigate(link_map[id]);
    }
    return (
        <div className="top-bar">
            <span className="logo">
                The Narrative Project
            </span>
            <span className='top-bar-nav-links'>
                <span className='top-bar-nav-link' onClick={() => onClickTopBarNavLink('home')}>Home</span>
                <span className='top-bar-nav-link' onClick={() => onClickTopBarNavLink('discover')}>Discover</span>
                <span className='top-bar-nav-link' onClick={() => onClickTopBarNavLink('content-explorer')}>Context Explorer</span>
            </span>
        </div>
    )
}
