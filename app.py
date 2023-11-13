from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
import pickle
import numpy as np

app = Flask(__name__)

from tensorflow.keras.models import load_model

model = load_model('intel_classifier_model_final.h5')


idx_to_classes = {
    0: 'buildings',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'
}
x = np.array(["buildings", "forest", "glacier", "mountain","sea","street"])
import matplotlib.pyplot as plt
import numpy as np
import time
def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(227, 227))
    img = img / 255
    images_list = [img]
    x1 = np.asarray(images_list)
    probability = model.predict(x1)
    
    
    probability = probability.flatten()
    probability = probability*100
    print(probability)

    plt.figure(figsize=(10,6))
    
    plt.barh(x, probability)
    plot_path = f'static/images/plot_{int(time.time())}.png'
    plt.savefig(plot_path)
    pred = np.argmax(probability)
    
    return plot_path
result = 'static/images/load2.gif'
@app.route('/')
def welcome():
    return render_template('index.html',result = result)

@app.route('/predict', methods=['POST'])
def pred():
    if request.method == 'POST':
        f = request.files['file']  # 'file' instead of 'filename'
        result = 'static/images/load.gif'
        if f:
            result = 'static/images/load.gif'
            file_path = os.path.join('uploads', secure_filename(f.filename))
            f.save(file_path)
            # Make prediction
            preds = model_predict(file_path, model)
            result = preds
            return render_template('index.html', result=result)
    return render_template('index.html',result= result)

if __name__ == '__main__':
    app.run(debug=True)
