from flask import Flask
from flask_cors import CORS
import logging

def create_app():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    app = Flask(__name__)
    CORS(app)
    
    # Register Blueprints / Routes
    from src.routes import main
    app.register_blueprint(main)
    
    return app
