#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import cv2
import os
from PIL import Image



class ResnetBlock(Model):
    def __init__(self,filters,strides=1,residual_path=False):
        super().__init__()
        self.filters=filters
        self.strides=strides
        self.residual_path=residual_path
        
        self.c1=Conv2D(filters,(3,3),strides=strides,padding='same',use_bias=False)
        self.b1=BatchNormalization()
        self.a1=Activation('relu')
        
        self.c2=Conv2D(filters,(3,3),strides=1,padding='same',use_bias=False)
        self.b2=BatchNormalization()
        
        if self.residual_path:
            self.down_c1=Conv2D(filters,(1,1),strides=strides,padding='same',use_bias=False)
            self.down_b1=BatchNormalization()
        self.a2=Activation('relu')
    def call(self,inputs):
        residual=inputs
        x=self.c1(inputs)
        x=self.b1(x)
        x=self.a1(x)
        x=self.c2(x)
        y=self.b2(x)       
        if self.residual_path:
            residual=self.down_c1(residual)
            residual=self.down_b1(residual)
        out=self.a2(y+residual)
        return out


class ResNet18(Model):
    def __init__(self):
        super().__init__()
        self.model=tf.keras.models.Sequential()
        self.model.add(Conv2D(64,(3,3),strides=1,padding='same',use_bias=False))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(ResnetBlock(filters=64,residual_path=False))
        self.model.add(ResnetBlock(filters=64,residual_path=False))
        self.model.add(ResnetBlock(filters=128,strides=2,residual_path=True))
        self.model.add(ResnetBlock(filters=128,residual_path=False))
        self.model.add(ResnetBlock(filters=256,strides=2,residual_path=True))
        self.model.add(ResnetBlock(filters=256,residual_path=False))
        self.model.add(ResnetBlock(filters=512,strides=2,residual_path=True))
        self.model.add(ResnetBlock(filters=512,residual_path=False))
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(36,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2()))
    def call(self,inputs):
        y=self.model(inputs)
        return y


model=ResNet18()


save_path='./save/save'
model.load_weights(save_path)#读入权重数据

output_path='./outputs/'
number=len(os.listdir(output_path))
pred_list="0123456789ABCDEFGHIGKLMNOPQRSTUVWXYZ"#标签对应的输出
print("检测结果：")
for i in range(0,number):#循环对每个分割出的字符进行喂入和输出
    img_arr=cv2.imread("./outputs/%d.jpg"%i)
    img_arr=Image.fromarray(img_arr.astype('uint8'))
    img_arr=img_arr.resize((16,16), Image.ANTIALIAS)
    img_arr=np.array(img_arr)
    img_arr=img_arr.astype(np.float)
    img_arr=img_arr/255.0
    img_arr=img_arr[None,...]
    pred=model.predict(img_arr)
    pred=tf.argmax(pred,axis=1)
    print(pred_list[pred.numpy()[0]],end='')





