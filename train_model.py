import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from helper_functions import *
from sklearn.model_selection import train_test_split
from model import get_model


def main():
    setup_gpu()
    image_files,mask_files = load_data()
    X_train,X_test,y_train,y_test = train_test_split(image_files,mask_files,test_size=0.1)
    
    model = get_model((256,256,3),2)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    train_sample = Generator(X_train,16)
    print('\n# Обучаем модель на тренировочных данных')
    history = model.fit(train_sample,steps_per_epoch = 64,epochs = 32)
    print('\nhistory dict:', history.history)

    model.save('wbc_segmentation_model.h5')

    test_sample = Generator(X_test,16)
    print('\n# Оцениваем на тестовых данных')
    results = model.evaluate(test_sample,batch_size = 32,steps = 64)
    print('\ntest loss, test acc:', results)

main()
