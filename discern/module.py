#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import os
import cv2

train_path="./data/train/"

def load_data(path):
    x_t=[]#训练集数据
    y_t=[]#训练集标签
    x_v=[]#测试集数据
    y_v=[]#测试集标签
    for i in range(0,36):
        names=os.listdir(path+str(i)+"/")
        count=0
        for name in names:
            if(count<800):#前800张作为训练集
                x_t.append(cv2.imread(path+str(i)+"/"+name))
                y_t.append([i])#文件夹的名字就是标签
            else:
                x_v.append(cv2.imread(path+str(i)+"/"+name))
                y_v.append([i])
            count=count+1
    x_t=np.array(x_t)
    y_t=np.array(y_t)
    y_t=y_t.astype(np.int64)
    x_v=np.array(x_v)
    y_v=np.array(y_v)
    y_v=y_v.astype(np.int64)    
    return x_t,y_t,x_v,y_v


x_train_save_path='./npy/x_train.npy'
y_train_save_path='./npy/y_train.npy'
x_test_save_path='./npy/x_test.npy'
y_test_save_path='./npy/y_test.npy'

if os.path.exists(x_train_save_path) and os.path.exists(y_train_save_path) and os.path.exists(
    x_test_save_path) and os.path.exists(y_test_save_path):#如果曾经读入过数据则直接读取
    print('-----------Load Data--------------')
    x_train=np.load(x_train_save_path)
    y_train=np.load(y_train_save_path)
    x_test=np.load(x_test_save_path)
    y_test=np.load(y_test_save_path)
else:
    print('-----------Generate Data--------------')
    x_train,y_train,x_test,y_test=load_data(train_path)#未读入过则生成数据
    print('-----------Save Data--------------')
    np.save(x_train_save_path,x_train)#将数据保存
    np.save(y_train_save_path,y_train)
    np.save(x_test_save_path,x_test)
    np.save(y_test_save_path,y_test)
    
x_train,x_test=x_train/255.0,x_test/255.0#图像归一化
np.random.seed(116)#打乱顺序
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

class ResnetBlock(Model):#ResNetBlock
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


model.compile(optimizer='adam',#配置反向传递优化设置
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])


check_point_save_path='./save/save'
if os.path.exists(check_point_save_path+'.index'):
    print('--------------Load The Model---------------')
    model.load_weights(check_point_save_path)#如果有训练过的数据直接读入
cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=check_point_save_path,
                                               save_weights_only=True,
                                               save_best_only=True)#设置断点续训
history=model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1,
                  callbacks=[cp_callback])#记录训练过程进行统计
model.summary()


file = open('./weights.txt', 'w')#保存权重
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)#绘制训练过程中准确率和误差变化
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

