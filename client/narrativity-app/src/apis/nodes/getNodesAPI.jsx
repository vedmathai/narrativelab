import axios from 'axios';


const url = 'http://localhost:5000'

export default async function getNodesAPI() {
    const nodes = await axios
        .get(url + '/v1/nodes', {
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            const nodes = response.data;
            return nodes
        })
        .catch((err) => console.log(err))
    return nodes
}
