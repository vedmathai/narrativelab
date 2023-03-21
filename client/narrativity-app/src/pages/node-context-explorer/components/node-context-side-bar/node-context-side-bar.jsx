
import './node-context-side-bar.css'
import 'common/side-bar/side-bar.css'


export default function NodeContextSideBar(props) {
    var properties = ""
    if (props.focussedOtherNodeID && props.nodeContext) {
        const node_id = props.focussedOtherNodeID;
        const relationship_id = props.nodeContext.node_id2relationship_id[node_id];
        var relationship = props.nodeContext.id2relationship[relationship_id];
        if (relationship) {
            properties = <div>
                <span
                    className="side-bar-relationship-property-key"
                >
                    Preposition:
                    
                </span>
                <span
                    className="side-bar-relationship-property-value"
                >
                    {relationship.preposition}
                </span>
            </div>
        }
    }
        
    return (
        <div className="side-bar node-context-side-bar">
            <div className="side-bar-content">
                <div className="side-bar-heading">
                    Relationship Properties
                </div>
                {properties}
            </div> 
        </div>
    )
}
