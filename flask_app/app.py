from flask import Flask
from text_generator.prediction import prediction_route

def create_app():
    app = Flask(__name__)
    app.register_blueprint(prediction_route)

    return app
