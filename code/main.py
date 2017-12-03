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


#CNN model

(actual_y, logit_output) = create_model()

#-------Cross entropy loss function------
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = actual_y, logits = logit_output))

#training using optimizers: AdamOptimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#gives a boolean vector for whether the actual and predicted output match (1 if true, 0 if false)
right_prediction = tf.equal(tf.argmax(logit_output, 1), tf.argmax(actual_y, 1))

#get accuracy
accuracy = tf.reduce_mean(tf.cast(right_prediction, tf.float32))


#train and evaluate the model
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# saver = tf.train.Saver()
	for i in range(20000):
		batch = train_data(50)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x: batch[0], actual_y: batch[1], no_drop_prob: 1.0})
			print("At step %d, training accuracy: %.2f" %(i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], actual_y: batch[1], no_drop_prob: 0.5})


# test = Image.open("../data/CelebA/Img/img_align_celeba/"+imageNames[52])
# img_array = np.asarray(test)
# test.show()

#"../../CelebA/Anno/list_attr_celeba.txt"
