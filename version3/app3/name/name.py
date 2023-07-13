from name.model import Backbone
import torch
from name.mtcnn import MTCNN
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
mtcnn = MTCNN()


threshold = 0.4

def get_img(img):
    face = mtcnn.align(img)
    transfroms = Compose(
        [ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return transfroms(face).unsqueeze(0)

def get_vector(img, model):
    img = get_img(img)
    vec = model(img)[0]
    return vec



def get_similarity(test_vector, data_vector):
    sim = test_vector.dot(data_vector)
    return sim



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


def get_name(img):
    data_name,data_vector = read_data()

    model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
    model.load_state_dict(torch.load('name/Pretrained_Models/model_ir_se50.pth',map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()

    test_vector = get_vector(img,model)

    similarity = []
    for vec in data_vector:
        similarity.append(get_similarity(test_vector, torch.tensor(vec)))


    max_similarity = max(similarity)
    max_index = similarity.index(max_similarity)
    name = data_name[max_index]
    if max_similarity < threshold:
        name = 'no one'

    return name



