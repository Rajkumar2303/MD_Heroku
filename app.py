import numpy as np
import requests
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from keras.preprocessing import image as keras_image
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from flask_cors import CORS


app = Flask(__name__)
#CORS(app)
# Load the LightGBM model
model = tf.keras.models.load_model('CMRmn2.h5')

# Define the label encoder or preprocessing steps if needed
label = {0: 'DCM', 1: 'HCM',2: 'MINF',3: 'NORMAL',4: 'RV'}




@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the JSON data from the request
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'})
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'})
            
            if file and allowed_file(file.filename):

                img = Image.open(file)
                # Preprocess the image
                img = img.resize((224, 224))
                if img.mode != 'RGB':
                    img=img.convert('RGB')
                else:
                    pass
   
                img_array = keras_image.img_to_array(img)
                
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0 
                tes_df = pd.DataFrame({'image_data': [img_array]})
                tes_datagen = ImageDataGenerator()
                img_data_array = np.array([tes_df['image_data'][0]])
                img_data_array = np.squeeze(img_data_array, axis=1)
                
                tes_gen = tes_datagen.flow(img_data_array, shuffle=False, batch_size=1)
                
                y_pred = model.predict(tes_gen)

                ind = int(np.argmax(y_pred, axis=1))
                        
                # Determine the predicted class
                predicted_class = label[ind]

                # Return the prediction result as JSON
                return jsonify({'predicted_Disease': predicted_class})
                
            else:
                return jsonify({'error': 'No image URL provided'})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Invalid request method'})


# Function to check if file extension is allowed
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True)
