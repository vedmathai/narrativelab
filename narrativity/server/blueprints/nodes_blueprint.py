
from flask import Blueprint, Response, request
from http import HTTPStatus
import json

from narrativity.server.server import NarrativityServer

nodes_blueprint = Blueprint('nodes_blueprint', __name__)


@nodes_blueprint.route('nodes/most-connected', methods=['GET'])
def get_most_connected_node():
    server = NarrativityServer.instance()
    most_connected_node = server.get_most_connected_node()
    return Response(
        json.dumps(most_connected_node.to_dict()),
        HTTPStatus.OK
    )

@nodes_blueprint.route('nodes/<node_id>/context', methods=['GET'])
def get_node_context(node_id):
    server = NarrativityServer.instance()
    node_context = server.get_node_context(node_id)
    return Response(
        json.dumps(node_context.to_dict()),
        HTTPStatus.OK
    )
