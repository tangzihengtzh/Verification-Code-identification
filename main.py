import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s)!")
    for i in range(device_count):
        print(torch.cuda.get_device_name(i))
else:
    print("No CUDA devices found.")

def my_collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    # 如果数据类型不是 tensor，则将其转换为 tensor
    if not isinstance(data[0], torch.Tensor):
        data = [torch.tensor(d) for d in data]
    # 如果目标类型不是 tensor，则将其转换为 tensor
    if not isinstance(target[0], torch.Tensor):
        target = [torch.tensor(t) for t in target]
    return [data, target]

captcha_list = list('0123456789abcdefghijklmnopqrstuvwxyz')
captcha_length = 5
def text2vec(text):
    vector = torch.zeros((captcha_length, len(captcha_list)))
    text_len = len(text)
    if text_len > captcha_length:
        raise ValueError("验证码过长")
    for i in range(text_len):
        vector[i,captcha_list.index(text[i])] = 1
    return vector

def vec2text(vec):
    label = torch.nn.functional.softmax(vec, dim =1)
    vec = torch.argmax(label, dim=1)
    for v in vec:
        text_list = [captcha_list[v] for v in vec]
    return ''.join(text_list)

# print(vec2text(text2vec('12345')))

class MyDataset(Dataset):

    def __init__(self,root_dir):
        self.root_dir=root_dir
        self.img_names=os.listdir(root_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path=os.path.join(self.root_dir,self.img_names[idx])
        img=Image.open(img_path)
        # print("img:",img)
        img_tensor=transforms.ToTensor()(img)
        label=self.img_names[idx][:-4]
        label_tensor=text2vec(label)
        return img_tensor,label_tensor

    def test_showdir(self):
        print(self.root_dir)

    def test_showitem(self,idx):
        img_path = os.path.join(self.root_dir, self.img_names[idx])
        mat=cv2.imread(img_path,1)
        cv2.imshow("test_showitem",mat)
        label = self.img_names[idx][:-4]
        # print("label:",label)
        print(self.__getitem__(idx)[0].shape)
        print(self.__getitem__(idx)[1].shape)
        cv2.waitKey(0)

def getimg2tensor(img_path):
    img = Image.open(img_path)
    img_tensor = transforms.ToTensor()(img)
    print("输入张量:",img_tensor.shape)
    return img_tensor

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=False)
        self.fc1 = nn.Linear(9600, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 180)


    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 9600)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



def train(epoch_nums):
    batch=64
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MyDataset(r"D:\pythonItem\yanzhengmashibie\my_dataset_RGB")
    train_data_loader = DataLoader(train_dataset, batch_size=batch, num_workers=0, shuffle=True, drop_last=True)
    # train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=my_collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('当前设备是:',device)
    Mymodel = MyCNN()
    Mymodel.to(device)

    criterion = nn.MultiLabelSoftMarginLoss() # 损失函数
    optimizer = torch.optim.Adam(Mymodel.parameters(), lr=0.001) # 优化器
    # 加载模型
    model_path = 'D:/pythonItem/savedmodel/MyCNN.pth'
    if os.path.exists(model_path):
        print('开始加载模型')
        checkpoint = torch.load(model_path)
        Mymodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    i = 1
    for epoch in range(epoch_nums):
        if(epoch%50==0):
            print("epoch:",epoch)
        running_loss = 0.0
        Mymodel.train() # 神经网络开启训练模式
        for data in train_data_loader:
            # print(data)
            # exit(2)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) #数据发送到指定设备
            #每次迭代都要把梯度置零
            optimizer.zero_grad()
            # 前向传播
            outputs = Mymodel(inputs).view(batch,5,36)
            # 计算误差
            loss = criterion(outputs, labels)
            # 后向传播
            loss.backward()
            # 优化参数
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 0:
                print("当前loss:",running_loss/200)
                # acc = calculat_acc(outputs, labels)
                # print('第%s次训练正确率: %.3f %%, loss: %.3f' % (i,acc,running_loss/200))
                running_loss = 0
                # 保存模型
                # torch.save(Mymodel.state_dict(), model_path)
                torch.save({
                            'model_state_dict':Mymodel.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            },model_path)
            i += 1
        # 每5个epoch 更新学习率
        if epoch % 5 == 4:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9

# train(2000)

Mymodel = MyCNN()
model_path=r"D:\pythonItem\savedmodel\MyCNN.pth"
checkpoint = torch.load(model_path)
Mymodel.load_state_dict(checkpoint['model_state_dict'])
# Mymodel.load_state_dict(torch.load(r"D:\pythonItem\savedmodel\MyCNN.pth"))


# out=Mymodel(getimg2tensor(r"D:\pythonItem\yanzhengmashibie\test.jpg"))
img_path=r"D:\pythonItem\yanzhengmashibie\my_dataset_RGB\n8ydd.jpg"
img_tensor=getimg2tensor(img_path)
img_tensor = img_tensor[0:3, :, :]
out=Mymodel(img_tensor)
print("输出张量:",out.shape)
out=out.view(-1, len(captcha_list))
print("识别为:",vec2text(out))
mat = cv2.imread(img_path, 1)
cv2.imshow("test", mat)
cv2.waitKey(0)



# Mymodel=MyCNN()
# opt=torch.optim.SGD(Mymodel.parameters(),lr=0.001)
# traindata=MyDataset(root_dir=r"D:\pythonItem\yanzhengmashibie\my_dataset_RGB")
#
# traindata.test_showitem(0)
# print(traindata[0])