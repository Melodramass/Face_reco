from model import Backbone
import torch
from PIL import Image
from mtcnn import MTCNN

from torchvision.transforms import Compose, ToTensor, Normalize

mtcnn = MTCNN()


def get_img(img_path, device):
    img = Image.open(img_path)
    face = mtcnn.align(img)
    transfroms = Compose(
        [ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return transfroms(face).to(device).unsqueeze(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_gsx = get_img('face_image/gsx.jpg', device)
img_lhy = get_img('face_image/lhy.jpg', device)
img_lq = get_img('face_image/lq.jpg', device)
img_wbl = get_img('face_image/wbl.jpg', device)
img_wlyq = get_img('face_image/wlyq.jpg', device)
img_wzb = get_img('face_image/wzb.jpg', device)



model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
model.load_state_dict(torch.load('Pretrained_Models/model_ir_se50.pth',map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.eval()
model.to(device)

emb1 = model(img_gsx)[0].tolist()
emb2 = model(img_lhy)[0].tolist()
emb3 = model(img_lq)[0].tolist()
emb4 = model(img_wbl)[0].tolist()
emb5 = model(img_wlyq)[0].tolist()
emb6 = model(img_wzb)[0].tolist()

with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('高晟鑫 ' + ' '.join(map(str, emb1)) + '\n')
    f.write('李鸿远 ' + ' '.join(map(str, emb2)) + '\n')
    f.write('李乾 ' + ' '.join(map(str, emb3)) + '\n')
    f.write('王博乐 ' + ' '.join(map(str, emb4)) + '\n')
    f.write('汪廖雅琪 ' + ' '.join(map(str, emb5)) + '\n')
    f.write('王展博 ' + ' '.join(map(str, emb6)) + '\n')

