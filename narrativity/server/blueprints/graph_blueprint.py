
from flask import Blueprint, Response, request
from http import HTTPStatus
import json

from narrativity.server.server import NarrativityServer

graph_blueprint = Blueprint('graph_blueprint', __name__)


@graph_blueprint.route('graph/search', methods=['POST'])
def graph_search():
    server = NarrativityServer.instance()
    search_request = request.json
    search_response = server.search_graph(search_request)
    return Response(
        json.dumps(search_response.to_dict()),
        HTTPStatus.OK
    )
