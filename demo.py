import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from helper_functions import *

def main(model_name='wbc_segmentation_model.h5',test_img_path='dataset/data2/010.png'):
    setup_gpu()
    model = load_model(model_name)
    test_img = cv2.imread(test_img_path)

    dim = (test_img.shape[1],test_img.shape[0])
    original_img = test_img
    image = prepare_img(test_img)
    ret = predict_and_process(original_img, image,model,dim)
    fig,axes = plt.subplots(2,1)
    axes[0].imshow(ret)
    axes[1].imshow(test_img)
    path = '000.png'
    cv2.imwrite(path, ret)

    plt.show(block=True)

main()
