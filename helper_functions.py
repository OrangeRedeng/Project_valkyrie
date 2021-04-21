import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import os
import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import random

def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate 3GB of memory on the first GPU
      try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

def create(pic,directory_path):
    images = []
    masks = []

    p_img = cv2.imread(directory_path+pic)
    name = pic.split('.')[0]

    p_img = cv2.imread(directory_path+name+ '.png')
    p_mask = cv2.imread(directory_path+name+'_2.png')
    
    p_img = cv2.resize(p_img,(256,256))
    p_mask = cv2.resize(p_mask,(256,256))
    
    p_mask = cv2.cvtColor(p_mask,cv2.COLOR_BGR2GRAY)
    p_mask[p_mask>0] = 1
    return p_img, p_mask

def load_data():
    data = os.listdir('dataset/data1/')
    image_files = []
    mask_files = []
    for i in data:
        if '_2.png' in i:
            mask_files.append(i)
        else:
            image_files.append(i)
    return image_files,mask_files

def Generator(X_list,batch_size):
    while 1:
        b = 0
        all_i = []
        all_m = []
        random.shuffle(X_list)
        for i in range(batch_size):

            image,mask = create(X_list[i-1],'dataset/data1/')
            all_i.append(image)
            mask = mask.reshape((256,256,1))
            all_m.append(mask)
            b+=1

        all_i = np.array(all_i)
        all_m = np.array(all_m)
        all_m = to_categorical(all_m)
        yield np.array(all_i),np.array(all_m)

def prepare_img(image):
    image = cv2.resize(image,(256,256))
    image = image.reshape((1,256,256,3))
    return image

def predict_and_process(original_image,image,model,dim):
    image = image.reshape((256,256,3))
    pred = model.predict(image.reshape(1,256,256,3))
    
    pred[pred>0.6] = 1
    pred[pred<0.5] = 0
    pred = pred[:,:,:,1].reshape((256,256))

    pred = cv2.resize(pred, dim, interpolation = cv2.INTER_AREA)
    pred = np.uint8(pred)
    original_image = np.uint8(original_image)
    ret = cv2.bitwise_and(original_image, original_image, mask = pred)

    return ret
