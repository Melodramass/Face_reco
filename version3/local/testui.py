import requests
import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk#图像控件

cap = cv2.VideoCapture(0)#创建摄像头对象
photo = None
img = None

def tkImage():
    ref,frame=cap.read()
    frame = cv2.flip(frame, 1) #摄像头翻转
    crop = frame[0:480,160:480]
    cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    pilImage=Image.fromarray(cvimage)
    pilImage = pilImage.crop([160,0,480,480])
    tkImage = ImageTk.PhotoImage(image=pilImage)
    return tkImage,crop

top = tk.Tk()
top.title('视频窗口')
top.geometry('1200x800')

detected_word = "未检测"
detected_text = Text(top, width=60, height=3, font=("楷体",20))
detected_text.insert(1.0, detected_word)
detected_text.place(x=200, y=600)

#界面画布更新图像
def fun():
    cv2.imwrite("output/test.png", crop)
    global photo
    global img
    global detected_text
    global new_img
    img = Image.open("output/test.png")
    img = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开
    original_img_label = Label(top, image=img)
    original_img_label.place(x=430, y=100)

    image = open('output/test.png','rb')
    
    #response anti
    response_anti = requests.post('http://172.22.1.120:8000/face_info', files={'image': image})
    #response1 = requests.post('http://localhost:5000/predict', files={'image': image})
    prediction_word = response_anti.json()['prediction_word']
    score = response_anti.json()['score']
    test_speed = response_anti.json()['test_speed']
    name = response_anti.json()['name']

    '''#response name
    response_name = requests.post('http://172.22.1.120:8000/name')
    name = response_name.json()['name']
    print('response_name',response_name.status_code)'''

    #response image
    response_image = requests.get('http://172.22.1.120:8000/download')
    #response2 = requests.get('http://localhost:5000/download')
    print('response_image',response_image.status_code)
    with open('processed_image_local.png', 'wb') as f:
        f.write(response_image.content)

    

    new_img = Image.open("processed_image_local.png")
    new_img = ImageTk.PhotoImage(new_img)  # 用PIL模块的PhotoImage打开
    new_image_label = Label(top, image=new_img)
    new_image_label.place(x=830, y=100)


    prediction_words = 'This is a ' + str(prediction_word) + '.' 
    score_words = 'The score is ' + str(score) + '.' + '\n'
    anti_words = prediction_words + score_words

    name_words = 'The person is ' + str(name) + '.' + '\n'

    test_speed_words = 'The test speed is ' + str(test_speed) + '.'

    detected_text.delete(1.0, END)
    detected_text.insert(1.0, anti_words+name_words+test_speed_words)

    
    


    

    
    

btn = Button(top, text="检测", command=fun)
btn.place(x=100, y=600)


canvas = Canvas(top,bg = 'white',width = 320,height = 480 )#绘制画布
Label(top,text = '实时监控',font = ("黑体",20),width =15,height = 1).place(x =100,y = 20,anchor = 'nw')
canvas.place(x = 50,y = 100)


while True:
    try: #关闭窗口后，由于没有采集图像帧，所以加载不到画布上，会弹出报错。采用异常跳出程序中止
        pic,crop = tkImage()
        canvas.create_image(0,0,anchor = 'nw',image = pic)
        top.update()
        top.after(1)
    except: #关闭窗口后执行
        cap.release() #释放摄像头
        top.mainloop() #关闭窗口