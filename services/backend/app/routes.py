from app import app, response
from app.controller import DataModelController
from flask import request
import os

@app.route('/')
def index():
    # return generate_password_hash(password='password')
    return 'Hello, World!'

@app.route('/data-model/data-test', methods=['POST'])
def upload():
    if request.method == 'POST':
        return DataModelController.store()
    else:
        return response.error(405, "Method Not Allowed")
