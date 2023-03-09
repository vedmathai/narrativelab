"""
    This is the main entrypoint and where the Server is instantiated.
    The blueprints to all the routes are registered here.
"""
import argparse
from flask import Flask
from flask_cors import CORS

from narrativity.server.blueprints.corpus_blueprint import corpus_blueprint
from narrativity.server.blueprints.nodes_blueprint import nodes_blueprint


from narrativity.server.server import NarrativityServer


if __name__ == '__main__':
    narrativity_app = Flask(__name__)
    CORS(narrativity_app)
    parser = argparse.ArgumentParser()
    NarrativityServer.instantiate(narrativity_app)

    # registering the blueprints
    blueprints = [
        corpus_blueprint, nodes_blueprint
    ]
    for blueprint in blueprints:
        narrativity_app.register_blueprint(blueprint, url_prefix='/v1')

    # starting the server
    narrativity_app.run(debug=True)
