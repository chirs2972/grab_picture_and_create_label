
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:12:33 2020

@author: opgg 
@updater: chris
"""

#import torch
#import torchvision
#import matplotlib.pyplot as plt

from PIL import Image 
import os
import numpy as np
import csv
import torchvision.transforms as transforms
import cv2
import shutil

#請設定Train與Test的圖片編號範圍
Trainfront=1
Trainrear=1100
Testfront=1201
Testrear=1400

#刪除測資與創建所需的資料夾
os.makedirs(r'./source_data/',exist_ok=True)
source = os.listdir(r"./source_data/")
if source != None:
	try:
		shutil.rmtree(r"data_img/mnist_train/")
		shutil.rmtree(r"data_img/mnist_test/")
	except:
		print('mnist train與test不存在, 進行創建程序')

os.makedirs(r'./data_img/mnist_train/',exist_ok=True)
os.makedirs(r'./data_img/mnist_test/',exist_ok=True)

path = os.getcwd()

#將source_data內的圖片移至mnist_train與mnist_test
for i in range(Trainfront,Trainrear+1):
	shutil.copyfile(path+"./source_data/"+str(i)+".png",path+"/data_img/mnist_train/"+str(i)+".png")
for i in range(Testfront,Testrear+1):
	shutil.copyfile(path+"./source_data/"+str(i)+".png", path+"/data_img/mnist_test/"+str(i)+".png")

train = os.listdir(r"data_img/mnist_train/")
test = os.listdir(r"data_img/mnist_test/")

#開啟名為grade的csv，並轉換成list
gfile = open('grade.csv','r')
grade = csv.reader(gfile)
grade = list(grade)




# 內插改變影像大小
trans = transforms.Resize((64, 64))

# 開啟輸出的 CSV 檔案
train_config_path = 'data_img/mnist_train.csv'
test_config_path = 'data_img/mnist_test.csv'

train_img_path = 'data_img/mnist_train/'
test_img_path = 'data_img/mnist_test/'


with open(train_config_path, 'w', newline='') as csvFile:
	writer = csv.writer(csvFile)
	for i in train:
		img = Image.open(train_img_path + str(i))
		img_path = "./" + train_img_path + str(i)
		img_num = os.path.splitext(i)[0]
		img_trans = trans(img)
		img_trans = np.asanyarray(img_trans)
		cv2.imwrite(img_path,img_trans)	

		#寫入檔案路徑與label至mnist_train.csv
		writer.writerow([img_path,grade[int(img_num)-1][1]])
		
with open(test_config_path, 'w', newline='') as csvFile:
	writer = csv.writer(csvFile)
	for i in test:
		img = Image.open(test_img_path + str(i))
		img_path = "./" + test_img_path + str(i)
		img_num = os.path.splitext(i)[0]
		img_trans = trans(img)
		img_trans = np.asanyarray(img_trans)
		cv2.imwrite(img_path,img_trans)	
		#寫入檔案路徑與label至mnist_test.csv
		writer.writerow([img_path,grade[int(img_num)-1][1]]) 		

#關閉csv檔案
gfile.close()
