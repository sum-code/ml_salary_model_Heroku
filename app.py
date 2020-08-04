# -*- coding: utf-8 -*-
import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

#first this will create web flask app
app=Flask(__name__)

#for reading model pickle open in read mode
model=pickle.load(open('model.pkl','rb'))

#this is my home page
#by deflult it will render my home app index.html
@app.route('/')
def home():
    return render_template('index.html')
    
    
#this is some like web api

@app.route('/predict',methods=['POST'])
#because of /predict it will go and hit the function
def predict():
    #here when we use request then we can use all values on that field
    #and store in int_feature list
    int_features=[int(x) for x in request.form.values()]
    #create array
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output=round(prediction[0],2)
    
    return render_template('index.html',prediction_text='Employee Salary should be ${}'.format(output))
#this is main function which will run hole flask
if __name__ == "__main__":
    app.run(debug=True)