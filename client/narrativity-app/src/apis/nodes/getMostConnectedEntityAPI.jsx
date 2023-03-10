import axios from 'axios';


const url = 'http://localhost:5000'

export default async function getMostConnectedNodeAPI() {
    const node = await axios
        .get(url + '/v1/nodes/most-connected', {
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            const node = response.data;
            return node
        })
        .catch((err) => console.log(err))
    return node
}
