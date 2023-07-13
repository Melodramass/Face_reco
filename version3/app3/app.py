from flask import Flask, request, jsonify, send_file
from PIL import Image
import os
import warnings

from name.name import get_name
from anti_spoofing.anti_spoofing import anti
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

warnings.filterwarnings('ignore')

app = Flask(__name__)


@app.route('/face_info', methods=['POST'])

def face_info():

    image_file = request.files['image']
    image = Image.open(image_file)
    
    start = time.time()

    prediction_word, value = anti(image)
    name_img = Image.open('images/unprocessed_image_app.png')
    person_name = get_name(name_img)

    test_speed = time.time() - start

    response = {}
    response['prediction_word'] = prediction_word
    response['score'] = value
    response['test_speed'] = test_speed
    response['name'] = person_name

    return jsonify(response)
    
'''@app.route('/name', methods=['POST'])

def name():
    name_img = Image.open('images/unprocessed_image_app.png')
    name = get_name(name_img)
    response = {}
    response['name'] = name
    
    return jsonify(response)'''

@app.route('/download')
def download():
    

    return send_file('images/processed_image_app.png', mimetype='image/png', as_attachment=True)


if __name__ == '__main__':
    app.run(host="172.22.1.120",port=8000)