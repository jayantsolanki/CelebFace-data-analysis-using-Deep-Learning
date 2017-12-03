#CSE574 | Project 4
#Description: TO implement a convolutional neural network to determine 
#whether the person in a portrait image is wearing glasses or not
#########################################################################

# import tensorflow as tf
import PIL.ImageOps
from PIL import Image
from sklearn import preprocessing
import numpy as np
from libs import *

length = 0;
labelnames = np.zeros(41)
filename = '../data/CelebA/Anno/list_attr_celeba.txt'
imagepath = '../data/CelebA/Img/img_align_celeba/'
celebData = 0
(celebData, labels, imageNames) = dataloader(imagepath, filename)
print("Celebdata dimension is ", celebData.shape)
print("labels dimension is ", labels.shape)
print("imageNames dimension is ", imageNames.shape)

#eyglasses at column 15+1
#Extract data

# for i in range(10):
# 	for file in os.listdir("../proj3_images/Numerals/"+str(i)+"/"):



test = Image.open("../data/CelebA/Img/img_align_celeba/"+imageNames[52])
img_array = np.asarray(test)
test.show()

#"../../CelebA/Anno/list_attr_celeba.txt"
