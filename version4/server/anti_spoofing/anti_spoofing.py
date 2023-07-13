"""
本文件用于识别对象是否为活体
"""
import cv2
import numpy as np
from anti_spoofing.src.anti_spoof_predict import AntiSpoofPredict
from anti_spoofing.src.generate_patches import CropImage
import os
from anti_spoofing.src.utility import parse_model_name

#检测对象是否为活体
def anti(image):
    #获取图片并剪裁
    numpy_image = np.asarray(image)
    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite( 'images/unprocessed_image_app.png', image)
    image = cv2.resize(image,(480,640))
    #对图像中人脸进行检测
    model_dir = "anti_spoofing/resources/anti_spoof_models"
    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    
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
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
    
    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    
    #结果展示
    if label == 1:
        prediction_word = 'Real Face'
        result_text = "Real Face Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        prediction_word = 'Fake Face'
        result_text = "Fake Face Score: {:.2f}".format(value)
        color = (0, 0, 255)

    #绘制矩形框
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    #标注图像
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)
    image = cv2.resize(image,(320,480))
    cv2.imwrite( 'images/processed_image_app.png', image)
    return prediction_word, value
