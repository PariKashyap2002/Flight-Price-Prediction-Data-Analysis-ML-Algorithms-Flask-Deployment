import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__,static_folder='static',static_url_path='/static')
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [(x) for x in request.form.values()]
    features = [np.array(float_features)]
    print(features)
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The price is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True, host= '0.0.0.0', port=5000)