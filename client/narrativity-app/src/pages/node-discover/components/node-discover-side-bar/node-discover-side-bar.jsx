
import './node-discover-side-bar.css'
import 'common/side-bar/side-bar.css'


export default function NodeDiscoverSideBar(props) {
    const isChecked = false;
    const handleOnChange = () => {

    };
        
    return (
        <div className="side-bar node-discover-side-bar">
            <div className="side-bar-content">
                <div className="side-bar-heading">
                    Search
                </div>
                <div className="node-discover-search">
                    <input className="node-discover-search-input"/>
                </div>
                <div className="node-discover-search-filters">
                    <div className="node-discover-search-filter">
                        <div className="node-discover-search-filter-checkbox">
                            <input
                                type="checkbox"
                                id="entities"
                                name="entities"
                                value="Entities"
                                checked={isChecked}
                                onChange={handleOnChange}
                            />
                            Entities
                        </div>

                        <div className="node-discover-search-filter-checkbox">
                            <input
                                type="checkbox"
                                id="narratives"
                                name="narratives"
                                value="Narratives"
                                checked={isChecked}
                                onChange={handleOnChange}
                            />
                            Narratives
                        </div>
                    </div>
                
                </div>
            </div> 
        </div>
    )
}
