#libs

import PIL.ImageOps
from PIL import Image
from sklearn import preprocessing
import numpy as np

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
			labels[count-3,0] = int(listValues[16])##store the eyeglasses param

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
