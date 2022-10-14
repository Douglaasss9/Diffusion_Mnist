## mnist_train mnist_test
import torch
import torchvision
from skimage import io
import os
mnist_train = torchvision.datasets.MNIST('./datasets_folder/MNIST_data', train=True, download=True)#首先下载数据集，并数据分割成训练集与数据集
mnist_test = torchvision.datasets.MNIST('./datasets_folder/MNIST_data', train=False, download=True)
diffusion_path = "./datasets_folder/mnist/train_diffusion"
os.makedirs(diffusion_path)
diffusion_step = 10
 
f=open("./datasets_folder/mnist/mnist_train.txt", 'w+')#在指定路径之下生成.txt文件
"""这个是对相同的数据保存在同一个文件夹下"""
for i, (img, label) in enumerate(mnist_train):
    img_path = "./datasets_folder/mnist/mnist_train"+"/"+str(label)
    if i % diffusion_step == 0:
        io.imsave(diffusion_path+"/" + str(i) + ".jpg", img)
    # 判断结果
    if not os.path.exists(img_path):
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(img_path)
 
    #/lib/python3.7/site-packages/skimage/io/_io.py中io.imsave将
    # if arr.dtype == bool改为if type(arr) == bool
    io.imsave(img_path+"/" + str(i) + ".jpg", img)#将图片数据以图片.jpg格式存在指定路径下
 
    f.write(str(label)+'，'+str(i)+".jpg\n")#将路径与标签组合成的字符串存在.txt文件下
f.close()#关闭文件

f = open("./datasets_folder/mnist/mnist_train.txt", 'w+')#在指定路径之下生成.txt文件
"""这个是对相同的数据保存在同一个文件夹下"""
for i, (img, label) in enumerate(mnist_test):
    img_path = r"./datasets_folder/mnist/mnist_test"+"/"+str(label)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    io.imsave(img_path+"/" + str(i) + ".jpg", img)#将图片数据以图片.jpg格式存在指定路径下
    f.write(str(label)+'，'+str(i)+".jpg\n")#将路径与标签组合成的字符串存在.txt文件下
f.close()#关闭文件