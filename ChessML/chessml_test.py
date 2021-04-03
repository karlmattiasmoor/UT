import cv2 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

from predict import predictboard

import time

start_time = time.time()

folder = ('C:\\Users\\Karl\\Desktop\\UT\\images\\New Folder\\timetest\\')

for filename in os.listdir(folder):
    print(filename)
    img = os.path.join(folder,filename)
    predictboard(img, filename)

print("--- %s seconds ---" % (time.time() - start_time))


