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
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

filename = '../data/CelebA/Anno/list_attr_celeba.txt'
imagepath = '../data/CelebA/Img/img_align_celeba/'
celebData = 0
(celebData, labels, imageNames) = dataloader2(imagepath, filename)

#(celebData, labels, imageNames) = dataloader(imagepath, filename)
# celebData = celebData[0:100000,:,:,:]
# celebData = celebData[0:100000,:,:]
# labels = labels[0:100000,:]
# imageNames = imageNames[0:100000]
print("Celebdata dimension is ", celebData.shape)
print("labels dimension is ", labels.shape)
print("imageNames dimension is ", imageNames.shape)
#exit()
# # view_image(normalized_X[9999,:,:], train_images_label[9999])
# trains_images = celebData[0:20000,:,:,:]#getting the training sets
# test_images = celebData[20000:25000,:,:,:]#getting the training sets
# train_images_labels = labels[0:20000,:]
# test_images_labels = labels[20000:25000,:]
# # trains_images = trains_images.reshape([20000,2475*3])#flattening the input array
# test_images = test_images.reshape([5000,2475*3])

# view_image(normalized_X[9999,:,:], train_images_label[9999])
train_num = 56000 #80% of 70000
val_num = 7000
test_num = 7000

trains_images = celebData[0:train_num,:,:]#getting the training sets
train_images_labels = labels[0:train_num,:]

val_images = celebData[train_num:train_num+val_num,:,:]#getting the validation sets
val_images_labels = labels[train_num:train_num+val_num,:]

test_images = celebData[train_num+val_num:train_num+val_num+test_num,:,:]#getting the training sets
test_images_labels = labels[train_num+val_num:train_num+val_num+test_num,:]

#flattening the input array
# trains_images = trains_images.reshape([20000,2475*3])
trains_images = trains_images.reshape([train_num,784])
val_images = val_images.reshape([val_num,784])
test_images = test_images.reshape([test_num,784])
train_images_labels = train_images_labels.reshape([train_num,])
val_images_labels = val_images_labels.reshape([val_num,])
test_images_labels = test_images_labels.reshape([test_num,])

#standardizing the image data set with zero mean and unit standard deviation
trains_images = preprocessing.scale(trains_images)
val_images = preprocessing.scale(val_images)
test_images = preprocessing.scale(test_images)

#creating one-hot vectors for labels
train_images_labels_mat = np.zeros((train_num, 2), dtype=np.uint8)
train_images_labels_mat[np.arange(train_num), train_images_labels.T] = 1
# train_images_labels = train_images_labels_mat
# print(train_images_labels[51,:])

val_images_labels_mat = np.zeros((val_num, 2), dtype=np.uint8)
val_images_labels_mat[np.arange(val_num), val_images_labels.T] = 1
# val_images_labels = val_images_labels_mat

test_images_labels_mat = np.zeros((test_num, 2), dtype=np.uint8)
test_images_labels_mat[np.arange(test_num), test_images_labels.T] = 1
# test_images_labels = test_images_labels_mat

print("Train images shape: ", trains_images.shape)
print("Train labels shape: ", train_images_labels.shape)
print("Test images shape: ", val_images.shape)
print("Test labels shape: ", val_images_labels.shape)
print("Test images shape: ", test_images.shape)
print("Test labels shape: ", test_images_labels.shape)

#eyglasses at column 15+1
#Extract data

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 2]
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  # onehot_labels = labels
  # loss = tf.losses.softmax_cross_entropy(
  #     onehot_labels=onehot_labels, logits=logits)
  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
  print(onehot_labels.shape)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Load training and eval data
# mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = np.asarray(trains_images, dtype=np.float32)  # Returns np.array
train_labels = train_images_labels
eval_data = np.asarray(val_images, dtype=np.float32)  # Returns np.array  # Returns np.array
eval_labels = val_images_labels
test_data = np.asarray(test_images, dtype=np.float32)  # Returns np.array  # Returns np.array
test_labels = test_images_labels

# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
  model_fn=cnn_model_fn, model_dir="tfcache/celeb_convnet_model")

# Set up logging for predictions
# Log the values in the "Softmax" tensor with label "probabilities"
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": train_data},
  y=train_labels,
  batch_size=100,
  num_epochs=3,
  shuffle=True)
mnist_classifier.train(
  input_fn=train_input_fn,
  steps=300,
  hooks=[logging_hook])

# Evaluate the validation set and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": eval_data},
  y=eval_labels,
  num_epochs=1,
  shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)

# Evaluate the validation set and print results
test_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": test_data},
  y=test_labels,
  num_epochs=1,
  shuffle=False)
test_results = mnist_classifier.evaluate(input_fn=test_input_fn)
print(test_results)



# for i in range(10):
# 	for file in os.listdir("../proj3_images/Numerals/"+str(i)+"/"):



#============================================================
'''
#CNN model
#Create nodes for input images and target labels
tf.set_random_seed(2103)
x = tf.placeholder(tf.float32, shape = [None, 784]) 
actual_y = tf.placeholder(tf.float32, shape = [None, 2]) 
#dropout only during training to prevent overfitting, not during testing
no_drop_prob = tf.placeholder(tf.float32)  #probability of not dropping out the neurons output

logit_output = create_cnn_model(x, actual_y, no_drop_prob)

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
	for i in range(101):
		data_batch, target_batch = tf.train.batch(dequeue_op, batch_size=15, capacity=40)
		# for j in np.arange(0, train_num-2000, 2000):
		# 	trainData = trains_images[j:j+2000,:]
		# 	trainLabels = train_images_labels[j:j+2000,:]
		# 	# print(i)
		# 	if (i%10 == 0 and j == 0):
		# 		train_accuracy = accuracy.eval(feed_dict={x: trainData, actual_y: trainLabels, no_drop_prob: 1.0})
		# 		print("At step %d, training accuracy: %.2f" %(i, train_accuracy))
		# 	train_step.run(feed_dict={x: trainData, actual_y: trainLabels, no_drop_prob: 0.5})

		# print("Epoch ",i)
		if(i%10==0):
			# _, loss_value = sess.run([train_step, cross_entropy], feed_dict={x: trainData, actual_y: trainLabels})
			print('At epoch %d, cross_entropy loss is %.2f' %(i+1,cross_entropy.eval(feed_dict={x: trainData, actual_y: trainLabels, no_drop_prob: 1.0})))
		
	#Run on validation data
	accuracy_val = accuracy.eval(feed_dict={x: val_images, actual_y: val_images_labels, no_drop_prob: 1.0})
	print("CelebA validation accuracy: %.2f" %(accuracy_val*100))

	#Run on test data
	accuracy_test = accuracy.eval(feed_dict={x: test_images, actual_y: test_images_labels, no_drop_prob: 1.0})
	print("CelebA test accuracy: %.2f" %(accuracy_test*100))

# test = Image.open("../data/CelebA/Img/img_align_celeba/"+imageNames[52])
# img_array = np.asarray(test)
# test.show()

#"../../CelebA/Anno/list_attr_celeba.txt"
'''
