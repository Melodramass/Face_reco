from flask import Flask, request, jsonify, send_file
import numpy as np
from PIL import Image
import os
import cv2
import numpy as np
import warnings
import time

from name import get_name
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

warnings.filterwarnings('ignore')



app = Flask(__name__)


@app.route('/predict', methods=['POST'])

def predict():

    image_file = request.files['image']
    image = Image.open(image_file)
    numpy_image = np.asarray(image)
    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite( 'unprocessed_image_app.png', image)
    image = cv2.resize(image,(480,640))
   
    model_dir = "./resources/anti_spoof_models"
    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start
    
    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    

    if label == 1:
        prediction_word = 'Real Face'
        result_text = "Real Face Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        prediction_word = 'Fake Face'
        result_text = "Fake Face Score: {:.2f}".format(value)
        color = (0, 0, 255)
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)
    image = cv2.resize(image,(320,480))
    cv2.imwrite( 'processed_image_app.png', image)

    name_img = Image.open('unprocessed_image_app.png')
    name = get_name(name_img)
    response = {}
    response['prediction_word'] = prediction_word
    response['score'] = value
    response['test_speed'] = test_speed
    response['name'] = name

    
    
    
    return jsonify(response)
    


@app.route('/download')
def download():
    

    return send_file('processed_image_app.png', mimetype='image/png', as_attachment=True)


if __name__ == '__main__':
    app.run()