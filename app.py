# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 02:04:55 2024

@author: rajku
"""




# coding=utf-8

import os

from io import BytesIO
import numpy as np
import requests
# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify

#from gevent.pywsgi import WSGIServer




import tensorflow as tf
from PIL import Image

# Define a flask app
app = Flask(__name__)

# Load your trained model

model = tf.keras.models.load_model('CMRmn2.h5')




def model_predict(img, model):
    
    # x = np.true_divide(img, 255)
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
        try:
            # Get the JSON data from the request
            data = request.get_json()
            # Extract the URL from the JSON data
            image_url = data.get('url')
            if image_url:
                # Fetch the image from the URL
                response = requests.get(image_url)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img = img.resize((224, 224))                
                    preds = model_predict(img, model)
                    result=preds
                    return jsonify({'prediction':result})
                    
                else:
                    return jsonify({'error': 'Failed to fetch image from the URL'})
            else:
                return jsonify({'error': 'No image URL provided'})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Invalid request method'})



if __name__ == '__main__':
    app.run(debug=True)
