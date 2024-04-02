# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 02:04:55 2024

@author: rajku
"""




# coding=utf-8

import os


import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify

#from gevent.pywsgi import WSGIServer




import tensorflow as tf


# Define a flask app
app = Flask(__name__)

# Load your trained model

model = tf.keras.models.load_model('CMRmn2.h5')




def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    
    x = np.expand_dims(x, axis=0)
   

   

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="DCM"
    elif preds==1:
        preds="HCM"
    elif preds==2:
        preds="MINF"
    elif preds==3:
        preds="NORMAL"
    else:
        preds="RV"
    
    
    return preds




@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']



        # Make prediction
        preds = model_predict(f, model)
        #preds='Brain'
        result=preds
        return jsonify({ 
            'prediction':result})
    return None



if __name__ == '__main__':
    app.run(debug=True)
