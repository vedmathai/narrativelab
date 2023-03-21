import axios from 'axios';


const url = 'http://localhost:5000'

export default async function searchGraphAPI(search_graph_request) {
    const response = await axios
        .post(url + '/v1/graph/search', search_graph_request, {
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            const project = response.data;
            return project
        })
        .catch((err) => console.log(err))
    return response
}
