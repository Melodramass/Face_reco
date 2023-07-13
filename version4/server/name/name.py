"""
本文件用于对象与数据库的匹配
"""
from name.model import Backbone
import torch
from name.mtcnn import MTCNN
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
mtcnn = MTCNN()


threshold = 0.4

def get_img(img):
    face = mtcnn.align(img)
    if face is None:
        return None
    transfroms = Compose([ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return transfroms(face).unsqueeze(0)

#获得其特征向量
def get_vector(img, model):
    img = get_img(img)
    if img is None:
        return None
    vec = model(img)[0]
    return vec


#比较特征向量的相关程度
def get_similarity(test_vector, data_vector):
    sim = test_vector.dot(data_vector)
    return sim


#数据读取
def read_data():
    file = open('data/output.txt', 'r', encoding='utf-8')
    lines = file.readlines()
    name = []
    vector = []
    for line in lines:
            person = line.strip().split(' ')
            name.append(person[0])
            
            vector.append(list(map(float,person[1:] )))
    return name,vector

#进行对象与数据库匹配
def get_name(img):
    data_name,data_vector = read_data()

    model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
    model.load_state_dict(torch.load('name/Pretrained_Models/model_ir_se50.pth',map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()

    test_vector = get_vector(img,model)
    if test_vector is None:
        return None

    similarity = []
    for vec in data_vector:
        similarity.append(get_similarity(test_vector, torch.tensor(vec)))


    max_similarity = max(similarity)
    max_index = similarity.index(max_similarity)
    name = data_name[max_index]
    if max_similarity < threshold:
        name = 'no one'

    return name



