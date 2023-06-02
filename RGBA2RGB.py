import os
import cv2
'''png格式常常是32位的RGBA格式，A代表透明度，
   光是更改图片后缀，不能改变图片的位数，
   需要在openCV中进行色彩空间的转换，
   将png格式的32位RGBA转为jpg格式的24位RGB'''

def convert2jpg(filename):                                      # 将彩色图转灰度图的函数
    img = cv2.imread(file_path+'/'+filename, 1)                 # 1是以彩色图方式去读
    jpg_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(out_path + '/' + filename, jpg_img)             # 保存在新文件夹下，且图名中加GRAY

file_path = r"D:\pythonItem\yanzhengmashibie\traindata"                         # 输入文件夹
# os.mkdir(r"D:\pythonItem\yanzhengmashibie\stars_JPG")                           # 建立新的目录
out_path =r"D:\pythonItem\yanzhengmashibie\stars_JPG"                           # 设置为新目录为输出文件夹

for filename in os.listdir(file_path):                          # 遍历输入路径，得到图片名
    print(filename)
    convert2jpg(filename)
