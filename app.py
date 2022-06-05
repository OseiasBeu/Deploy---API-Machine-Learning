# importar os pacotes necessarios
import os
import numpy as np
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from tensorflow import keras

# instanciar Flask object
app = Flask(__name__)
# API object
api = Api(app)

# carregar meu model
model = keras.models.load_model('model/boston_housing_model.h5')


class HelloWorld(Resource):
    def get(self):
        return {'Nome': 'Carlos Melo'}

    def post(self):
        args = request.get_json(force=True)
        input_values = np.asarray(list(args['valores'].values())).reshape(1, -1)
        predicted = model.predict(input_values)[0]

        return jsonify({'previsao': float(predicted)})


# adicionar a API
api.add_resource(HelloWorld, '/')


if __name__ == '__main__':
    app.run()
