from flask import Flask, jsonify, request
import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
# import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

# Create dataframe from dictionary for imagepath
data = {
    'image_path': ['inf_image.jpg'],
    'label': ['no_label']
}
inf_df = pd.DataFrame(data)

# Create label indices and flip indices
indices = {
    'Apparel, Men': 0,
    'Apparel, Women': 1,
    'Footwear, Men': 2,
    'Footwear, Women': 3
}
flip_indices = dict((value, key) for key, value in indices.items())

# Create variable declaration
img_height = 160
img_width = 160
SEED = 1
BATCH = 64
np.random.seed(7)
# tf.random.set_seed(7)

best_model = tf.keras.models.load_model('improved_model_multiclass.hdf5')

@app.route("/")
def hello_world():
    return jsonify(data)

@app.route("/fashion", methods=['GET', 'POST'])
def data_inference():
    if request.method == 'POST':
        data = request.json
        img_list = data['image_array']

        # Save Model in Current directory
        path = ''
        cv2.imwrite(os.path.join(path , 'inf_image.jpg'), np.array(img_list))
                
        # Transform data inference from path from dataframe to new data
        ds_inf = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        dataframe=inf_df,
        x_col='image_path',
        y_col='label',
        target_size=(img_height, img_width),
        class_mode='categorical',
        batch_size=BATCH,
        shuffle=False,
        seed=SEED)

        # Model prediction 
        y_pred_inf_proba = best_model.predict(ds_inf) # Result in predict proba
        y_pred_inf = np.argmax(y_pred_inf_proba, axis=1)[0] # Find indices number which is max predict proba
        y_pred_inf = flip_indices[y_pred_inf] # convert to indices name

        response = {
            'code':200, 
            'status':'OK',
            'prediction': y_pred_inf,
            'predict_proba': float(y_pred_inf_proba.max())
        }

        os.remove('inf_image.jpg')
        
        return jsonify(response)
    return "Silahkan gunakan method post untuk mengakses model"

app.run(debug=True)
#------------------------------------------------------------------------------------------------------------