
from flask import Blueprint, Response, request
from http import HTTPStatus
import json

from narrativity.server.server import NarrativityServer

nodes_blueprint = Blueprint('nodes_blueprint', __name__)


@nodes_blueprint.route('nodes', methods=['GET'])
def get_nodes():
    server = NarrativityServer.instance()
    nodes = server.get_nodes()
    return Response(
        json.dumps(nodes),
        HTTPStatus.OK
    )
