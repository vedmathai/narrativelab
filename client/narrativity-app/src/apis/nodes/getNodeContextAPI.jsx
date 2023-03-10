import axios from 'axios';


const url = 'http://localhost:5000'

export default async function getNodeContextAPI(node_id) {
    const node_context = await axios
        .get(url + '/v1/nodes/' + node_id + '/context', {
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            const node_context = response.data;
            return node_context
        })
        .catch((err) => console.log(err))
    return node_context
}
