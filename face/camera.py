import cv2
import torch

import torchvision
import model as models
from PIL import Image
import util
import Config
import numpy as np
import matplotlib.pyplot as plt
import os


model = models.Point68()  # 实例化全连接层
if os.path.isfile(Config.MODEL_SAVE_PATH):
    print("loading ...")
    state = torch.load(Config.MODEL_SAVE_PATH)
    model.load_state_dict(state["net"])
    start_epoch = state["epoch"]
    print("loading over")
model.eval()

torch.set_default_tensor_type(torch.DoubleTensor)


transforms = torchvision.transforms.Compose(
    [
        # torchvision.transforms.Grayscale(),  # 转灰度图
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ]
)  # 定义图像变换以符合网络输入

emotion = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]  # 表情标签


cap = cv2.VideoCapture(-1)  # 摄像头，0是笔记本自带摄像头
face_cascade = cv2.CascadeClassifier(
    "/home/xws/Downloads/opencv-4.5.0/data/haarcascades/haarcascade_frontalface_default.xml"
)

point_size = 1
point_color = (0, 0, 255)  # BGR
thickness = 4  # 可以为 0 、4、8


# opencv自带的一个面部识别分类器
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[:, ::-1, :]  # 水平翻转，符合自拍习惯
    frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.1, 3)
    img = frame
    img_ = np.rot90(img, -1).copy()
    img_ = np.rot90(img_, -1).copy()
    img_ = np.rot90(img_, -1).copy()
    img_ = np.rot90(img_, -1).copy()
    # if len(face) >= 1:
    #     (x, y, w, h) = face[0]
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     img = frame[:][y : y + h, x : x + w]
    # 如果分类器能捕捉到人脸，就对其进行剪裁送入网络，否则就将整张图片送入
    # img = img.astype(np.uint8)

    image = Image.fromarray(img)
    # img.show()

    tensor = transforms(image)

    tensor1 = tensor.reshape(1, 3, 224, 224)
    pre = model(tensor1)

    # util.show(plt,tensor,pre)
    points = util.tensorToPoint(pre.detach())
    for p in points:
        p[0] = p[0] * Config.IMAGE_SIZE
        p[1] = p[1] * Config.IMAGE_SIZE
        cv2.circle(img_, (int(p[0]), int(p[1])), point_size, point_color, thickness)
        # cv2.rectangle(img_, (int(-10), int(0)), (int(200), int(300)), (0, 255, 0))
    # frame = cv2.putText(
    #     frame, emotion[0], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2
    # )
    # 显示窗口第一个参数是窗口名，第二个参数是内容
    cv2.imshow("emotion", img_)
    if cv2.waitKey(1) == ord("q"):  # 按q退出
        break
cap.release()
cv2.destroyAllWindows()