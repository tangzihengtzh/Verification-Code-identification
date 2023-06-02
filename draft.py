# import os
#
# s="abcde.jpg"
#
# label=s[:-4]
#
# print(label)


# from torchsummary import summary
# from torchvision.models import vgg16  # 以 vgg16 为例
#
# myNet = vgg16()  # 实例化网络，可以换成自己的网络
# summary(myNet, (3, 64, 64))  # 输出网络结构
import torch

a=torch.randn(4,2,2)
print(a)
a=a[1:4,:,:]
print(a)