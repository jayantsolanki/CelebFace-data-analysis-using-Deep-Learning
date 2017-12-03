#libs

import PIL.ImageOps
from PIL import Image
from sklearn import preprocessing
import numpy as np
# import tensorflow as tf

def dataloader(imagespath,filename):
	count = 0;
	length = 0
	try:
		data = np.load("CelebA.npz")
		labels = data['labels']
		imageNames = data['imageNames']
		imageData = data['imageData']
	except FileNotFoundError:
		labelnames = np.zeros(41)
		# listValues  = np.zeros(41)
		for line in open(filename):
			count = count+1
			if(count==1):
				length = int(line)#total count of the data value rows
				# break
			elif(count==2):
				labelnames = line.split(" ")#store the label names
				break
		print(length)
		labels = np.zeros((length,1), dtype=np.int)
		#178x218
		# imageNames = np.array((length,1), dtype=object)
		imageNames = ["" for x in range(length)]
		imageData = np.zeros((100001,55,45,3), dtype=np.uint8)
		# imageData = np.zeros((100001,218,178,3), dtype=np.uint8)
		count = 0
		print('....')
		for line in open(filename):
			count=count+1
			if(count==1):
				continue
			if(count == 2):
				continue
			listValues = line.split(" ")#this will store the data values
			listValues = list(filter(None, listValues))#use this function only for python3
			listValues[-1] = listValues[-1].strip()
			imageNames[count-3] = listValues[0]#store the filename
			if(int(listValues[16])==1):
				labels[count-3,0] = 1##store the eyeglasses param
			else:
				labels[count-3,0] = 0##store the eyeglasses param

			#getting individual image data
			imagesPath = imagespath+imageNames[count-3]#full image path
			image = Image.open(imagesPath)
			# new_image = make_square(test_image)
			# image = new_image.convert('L')
			# image = PIL.ImageOps.invert(image)
			image = image.resize((45, 55), Image.BICUBIC)
			img_array = np.asarray(image)
			imageData[count-3,:,:,:] = img_array
			# if(listValues[16]=='1'):
			# 	print(listValues[0])
			# 	print(labelnames[16])
			# 	imagespath = imagespath+listValues[0]
			# 	test = Image.open(imagespath)
			# 	img_array = np.asarray(test)
			# 	test.show()
			if((count-3)%1000==0):
				print(count-3)
				if(count-3==100000):
					break
		imageData = np.asarray(imageData)
		imageNames = np.asarray(imageNames)
		np.savez("CelebA.npz", imageNames=imageNames, labels=labels, imageData=imageData)
	# print("Data set is ",listValues)#debugging
	# print("Data set count is ",len(listValues))#debugging
	# print("Total lines are", count)
	# print("Label first 53 are", label[0,:])
	# print("Image name is", imageNames[0])
	# print("Image dimension is", imageData.shape)
	# # imagespath = imagespath+imageNames[0]
	return(imageData, labels, imageNames)

def create_cnn_model(x, actual_y, no_drop_prob):
	#Intialization of weights and bias
	W1 = weight_init([5, 5, 3, 32])
	bias1 = bias_init([32])

	#reshape data into 4d
	x_4d = tf.reshape(x, [-1, 45, 55, 3])

	#convolve the image, add bias, apply relu activation function
	conv1_output = tf.nn.relu(convolution(x_4d, W1) + bias1)

	#apply max pooling
	pool1_output = maxpool(conv1_output)

	#-----------Convolution layer 2 with 64 features applied on 5x5 patch of image-----------

	#Intialization of weights and bias
	W2 = weight_init([5, 5, 32, 64])
	bias2 = bias_init([64])

	#convolve the layer 1 output, add bias, apply relu activation function
	conv2_output = tf.nn.relu(convolution(pool1_output, W2) + bias2)

	#apply max pooling
	pool2_output = maxpool(conv2_output)

	#------------Fully Connected Layer with 1024 neurons--------------
	#Intialization of weights and bias
	W_fc = weight_init([7*7*64, 1024])
	bias_fc = bias_init([1024])

	#flatten the pool2 output, multiply it with weights, add bias and then apply relu function
	pool2_flat = tf.reshape(pool2_output, [-1, 7*7*64])

	fc1_output = tf.nn.relu(tf.matmul(pool2_flat, W_fc) + bias_fc)

	fc1_output_drop = tf.nn.dropout(fc1_output, no_drop_prob)

	#------------Logit Layer--------------
	W_logit = weight_init([1024, 2])
	bias_logit = bias_init([2])

	logit_output = tf.matmul(fc1_output_drop, W_logit) + bias_logit

	return logit_output


#Weight Variable initialization
def weight_init(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

#Bias Variable initialization
def bias_init(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#Function to convolve input with given weights, stride=1 and zero padding
def convolution(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

#Function for 2D max pooling
def maxpool(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')