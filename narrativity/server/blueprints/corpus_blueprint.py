
from flask import Blueprint, Response, request
from http import HTTPStatus
import json

from narrativity.server.server import NarrativityServer

corpus_blueprint = Blueprint('corpus_blueprint', __name__)


@corpus_blueprint.route('corpus', methods=['POST'])
def post_corpus():
    server = NarrativityServer.instance()
    corpus = request.json
    server.upload_corpus(corpus)
    return Response(
        json.dumps({'msg': 'Corpus uploaded successfully'}),
        HTTPStatus.OK
    )
