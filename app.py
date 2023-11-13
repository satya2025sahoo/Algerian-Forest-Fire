import joblib
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

ridge_model=joblib.load('ridge.pkl')
scaler=joblib.load('scaler.pkl')

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature= float(request.form.get('Temperature')) 
        RH= float(request.form.get('RH')) 
        Ws= float(request.form.get('Ws')) 
        Rain= float(request.form.get('Rain')) 
        FFMC= float(request.form.get('FFMC')) 
        DMC= float(request.form.get('DMC')) 
        ISI= float(request.form.get('ISI')) 
        Classes= float(request.form.get('Classes')) 
        Region= float(request.form.get('Region')) 
        
        new_data=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data)

        return render_template('get_data.html',result=result[0])
    else:
        return render_template('get_data.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
