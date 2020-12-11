import matplotlib.pyplot as plt

from PIL import Image
import util

plt.ion()
# #读取图像到数组中
# im = array(Image.open('/home/xws/Downloads/boll.jpeg'))
f = plt.figure()
# #绘制图像
t = util.imageToTensor("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_001.png")
img = util.tensorToImage(t[0])
# img = plt.imread("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_300.png")
plt.imshow(img)
width = t[1]
height = t[2]

# text = util.readText("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_300.pts")
points = util.textToPoint("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_001.pts")
# print("points = ", points)
points = util.pointToTensor(points)
# print("pointToTensor = ", points)
points = util.tensorToPoint(points)
# print("tensorToPoint = ", points)


# # 使用红色星状物标记绘制点
for p in points:
    plt.plot(p[0], p[1], "r+")


# #绘制前两个点的线
# plt.plot(x[:2], y[:2])

# #添加标题，显示绘制的图像
plt.title('Plotting:"pic1.png"')
plt.ioff()
plt.savefig("/home/xws/Downloads/python/python/face/test.png")
# plt.show()


print("this is end")

# load ibug a img and show the point
