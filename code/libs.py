#libs

import PIL.ImageOps
from PIL import Image
from sklearn import preprocessing
import numpy as np

def dataloader(imagespath,filename):
	count = 0;
	labelnames = np.zeros(41)
	for line in open(filename):
		count=count+1
		if(count==1):
			length = int(line)#total count of the data value rows
			# break
		elif(count==2):
			labelnames = line.split(" ")#store the label names
		else:
			listValues = line.split(" ")#this will store the data values
			listValues = list(filter(None, listValues))#use this function only for python3
			break

	print("Data set is ",listValues)#debugging
	print("Data set count is ",len(listValues))#debugging
	return(labelnames)


	
	return (x, y)