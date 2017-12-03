#########################################################################
#CSE574 | Project 4
#Description: TO implement a convolutional neural network to determine 
#whether the person in a portrait image is wearing glasses or not
#########################################################################

import tensorflow as tf
import PIL.ImageOps
from PIL import Image
from sklearn import preprocessing
import numpy as np

#Extract data

# for i in range(10):
# 	for file in os.listdir("../proj3_images/Numerals/"+str(i)+"/"):



test = Image.open("../../CelebA/Img/img_align_celeba/000001.jpg")
img_array = np.asarray(test)

#"../../CelebA/Anno/list_attr_celeba.txt"