import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd 
from waitress import serve
app.static_folder = 'static'
app= Flask(__name__)
## Loading the model
model=pickle.load(open('crop_recommender_model.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.fit_transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.fit_transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html", Crop="{}".format(output))

    app.run(debug=True)
mode="dev" 
if __name__ == "__main__":
    if mode=="dev":
        app.run(host='0.0.0.0' , port=5000, debug=True)
    else:
        serve(app, host='127.0.0.1', port=5000, threads=1)

