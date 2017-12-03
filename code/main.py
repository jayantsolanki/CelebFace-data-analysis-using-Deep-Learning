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
celebData = celebData[0:100000,:,:,:]
labels = labels[0:100000,:]
imageNames = imageNames[0:100000]
print("Celebdata dimension is ", celebData.shape)
print("labels dimension is ", labels.shape)
print("imageNames dimension is ", imageNames.shape)

# view_image(normalized_X[9999,:,:], train_images_label[9999])
trains_images = celebData[0:20000,:,:,:]#getting the training sets
test_images = celebData[20000:25000,:,:,:]#getting the training sets
train_images_labels = labels[0:20000,:]
test_images_labels = labels[20000:25000,:]
trains_images = trains_images.reshape([20000,2475*3])#flattening the input array
test_images = test_images.reshape([5000,2475*3])


trains_images = preprocessing.scale(trains_images)#standardising the image data set with zero mean and unit standard deviation
test_images = preprocessing.scale(test_images)

#eyglasses at column 15+1
#Extract data

# for i in range(10):
# 	for file in os.listdir("../proj3_images/Numerals/"+str(i)+"/"):



test = Image.open("../data/CelebA/Img/img_align_celeba/"+imageNames[52])
img_array = np.asarray(test)
test.show()

#"../../CelebA/Anno/list_attr_celeba.txt"
