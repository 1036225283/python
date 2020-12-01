import matplotlib.pyplot as plt

from PIL import Image
import util 


# #读取图像到数组中
# im = array(Image.open('/home/xws/Downloads/boll.jpeg'))

# #绘制图像
# imshow(im)

img = plt.imread("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_300.png")
plt.imshow(img)

text =util.readText("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_300.pts")
points = util.textToPoint(text)



# # 使用红色星状物标记绘制点
for p in points:
    plt.plot(p[0],p[1],"r_")



# #绘制前两个点的线
# plt.plot(x[:2], y[:2])

# #添加标题，显示绘制的图像
plt.title('Plotting:"pic1.png"')

plt.show()

# load ibug a img and show the point

path = "/home/xws/Downloads/300w_cropped/01_Indoor/"

name = "indoor_300.png"



