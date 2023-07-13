"""
本文件设置了项目前端
"""
import cv2
from PIL import Image,ImageTk
import tkinter
import requests
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk#图像控件

#创建一个TK界面
root = tkinter.Tk()
root.geometry("1200x750+50+10")
root.resizable(False, False)
root.title('人脸检测')

video=cv2.VideoCapture(0)

imgnew = 1


menubar = tkinter.Menu(root)
root.config(menu=menubar)
#布局UI界面
set_size = (1200, 750)
new_img1 = Image.open("bgimg/sun.jpg")
new_img1 = new_img1.resize(set_size)
new_photo1 = ImageTk.PhotoImage(new_img1)
imglabel1 = tkinter.Label(root, image=new_photo1)
imglabel1.place(x=0, y=0, width=1200, height=750)

# global new_photo
# global photo_back

set_size = (320, 480)
img222 = Image.open("bgimg/space.jpg")
img222 = img222.resize(set_size)
new_photo = ImageTk.PhotoImage(img222)
photo_back = new_photo


imglabel = tkinter.Label(root, text=' ', width=320, height=480, image=new_photo)
imglabel.place(x=440, y=125, width=320, height=480)

imglabel2 = tkinter.Label(root, text=' ', width=320, height=480, image=photo_back)
imglabel2.place(x=800, y=125, width=320, height=480)

result1 = tkinter.Text(root, width=30, height=3, font=("Times", 22, "bold"))
result1.place(x=665, y=630)
str1 = ""
result1.insert(1.0, str1)
#UI使用函数，并实现对后端的请求
def screenshot():
    global imgnew
    cv2.imwrite("output/newtest.png", imgnew)
    global new_photo
    global new_img
    global img_back
    global photo_back
    newsize = (320, 480)
    new_img = Image.open("output/newtest.png")
    new_img = new_img.resize(newsize)
    new_photo = ImageTk.PhotoImage(new_img)
    imglabel = tkinter.Label(root, width=320, height=480, image=new_photo)
    imglabel.place(x=440, y=125, width=320, height=480)

    image = open('output/newtest.png','rb')
    #上传截取到的图片，并接受返回的参数
    #response anti
    response_face = requests.post('http://172.22.1.120:8000/face_info', files={'image': image})
    #response1 = requests.post('http://localhost:5000/predict', files={'image': image})
    prediction_word = response_face.json()['prediction_word']
    score = response_face.json()['score']
    test_speed = response_face.json()['test_speed']
    name = response_face.json()['name']

    

    #emotion = response_face.json()['emotion']

    '''#response name
    response_name = requests.post('http://172.22.1.120:8000/name')
    name = response_name.json()['name']
    print('response_name',response_name.status_code)'''
    #从后端下载标注好的图像
    #response image
    response_image = requests.get('http://172.22.1.120:8000/download')
    #response2 = requests.get('http://localhost:5000/download')
    #print('response_image',response_image.status_code)
    with open('output/processed_image_local.png', 'wb') as f:
        f.write(response_image.content)

    #输入展示信息
    prediction_words = 'This is a ' + str(prediction_word) + '.' 
    score_words = 'The score is ' + str(score)[0:4] + '.' + '\n'
    first_words = prediction_words + score_words

    name_words = 'The person is ' + str(name) + '.' + '\n'
    if name == "not found":
        first_words = ''
    #emotion_words = 'The person is ' + str(emotion) + '.' + '\n'
    second_words = name_words #+ emotion_words

    third_words = 'The test speed is ' + str(test_speed)[0:4] + 's.'

    #detected_text.delete(1.0, END)
    #detected_text.insert(1.0, first_words+second_words+third_words)


    #UI界面上展示返回的已标注图片
    img_back = Image.open("output/processed_image_local.png")
    img_back = img_back.resize(newsize)
    photo_back = ImageTk.PhotoImage(img_back)
    imglabel2 = tkinter.Label(root, width=320, height=480, image=photo_back)
    imglabel2.place(x=800, y=125, width=320, height=480)

    global str1
    str1 = first_words+second_words+third_words
    result1 = tkinter.Text(root, width=30, height=3, font=("Times",22 , "bold"))
    result1.place(x=665, y=630)
    result1.delete(1.0, tkinter.END)
    result1.insert(1.0, str1)

btn01 = tkinter.Button(root, text=" 检 测 ", font=("heiti", 20), command=screenshot)
btn01.place(x=158, y=650)

#开启摄像头并持续监控
def imshow():
    global video
    global root
    global image
    global imgnew
    global img
    res,img=video.read()

    if res==True:
        global imgnew

        img = cv2.flip(img, 1)  # 摄像头翻转
        # imgnew = img
        crop = img[0:480, 160:480]
        imgnew = crop
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        #将adarray转化为image
        img = Image.fromarray(img)
        # 裁剪
        img = img.crop([160, 0, 480, 480])
        #显示图片到label
        img = ImageTk.PhotoImage(img)
        image.image=img
        image['image']=img
    #创建一个定时器，每10ms进入一次函数
    root.after(10,imshow)

#创建label标签
image = tkinter.Label(root, text=' ', width=320, height=480)
image.place(x=80, y=125, width=320, height=480)
#设置相关按钮退出
def exit01():
    root.destroy()
    video.release()

btn02 = tkinter.Button(root, text=' 退 出 ', font=("heiti", 20), command=exit01)
btn02.place(x=420, y=650)

label1 = tkinter.Label(root, text=" 实 时 监 控 ", font=("heiti", 30, "bold"))
label1.place(x=110, y=40)

label2 = tkinter.Label(root, text=" 待 测 画 面 ", font=("heiti", 30, "bold"))
label2.place(x=465, y=40)

label3 = tkinter.Label(root, text=" 检 测 结 果 ", font=("heiti", 30, "bold"))
label3.place(x=822, y=40)

#有背景图片的UI界面设置，以三张图片为例
def changecolor(color):
    if color == "blue":
        filename = "bgimg/sea.jpg"
    elif color == "white":
        filename = "bgimg/sun.jpg"
    elif color == "black":
        filename = "bgimg/sun1.jpg"
    global new_photo1
    global new_img1
    global imglabel1
    global new_photo
    global new_img
    global video
    global root
    global image
    global imgnew
    global img
    global str1
    global img_back
    global photo_back
    set_size = (1200, 750)
    new_img1 = Image.open(filename)
    new_img1 = new_img1.resize(set_size)
    new_photo1 = ImageTk.PhotoImage(new_img1)
    imglabel1 = tkinter.Label(root, image=new_photo1)
    imglabel1.place(x=0, y=0, width=1200, height=750)
    btn01 = tkinter.Button(root, text=" 检 测 ", font=("heiti", 20), command=screenshot)
    btn01.place(x=158, y=650)
    image = tkinter.Label(root, text=' ', width=320, height=480)
    image.image = img
    image['image'] = img
    image.place(x=80, y=125, width=320, height=480)
    imglabel = tkinter.Label(root, text=' ', width=320, height=480, image=new_photo)
    imglabel.place(x=440, y=125, width=320, height=480)
    btn02 = tkinter.Button(root, text=' 退 出 ', font=("heiti", 20), command=exit01)
    btn02.place(x=420, y=650)
    label1 = tkinter.Label(root, text=" 实 时 监 控 ", font=("heiti", 30, "bold"))
    label1.place(x=110, y=40)
    label2 = tkinter.Label(root, text=" 待 测 画 面 ", font=("heiti", 30, "bold"))
    label2.place(x=465, y=40)
    label3 = tkinter.Label(root, text=" 检 测 结 果 ", font=("heiti", 30, "bold"))
    label3.place(x=822, y=40)
    result1 = tkinter.Text(root, width=30, height=3, font=("Times", 22, "bold"))
    result1.place(x=665, y=630)
    result1.insert(1.0, str1)
    imglabel2 = tkinter.Label(root, width=320, height=480, image=photo_back)
    imglabel2.place(x=800, y=125, width=320, height=480)

#更换UI背景菜单
menu1 = tkinter.Menu(menubar, tearoff=False)
for item in ["blue", "white", "black"]:
    if item == "blue":
        menu1.add_command(label=item, command=lambda: changecolor(color="blue"))
    elif item == "white":
        menu1.add_command(label=item, command=lambda: changecolor(color="white"))
    elif item == "black":
        menu1.add_command(label=item, command=lambda: changecolor(color="black"))

menubar.add_cascade(label="yanse", menu=menu1)

imshow()

root.mainloop()

#释放video资源
video.release()

