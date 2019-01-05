![University at Buffalo](http://www.nsm.buffalo.edu/Research/mcyulab//img/2-line_blue_gray.png)

***
# CelebFace data analysis using a Convolution Neural Network

## Introduction
***
In this project we perfomed machine learning analysis using CNN the Celeb dataset which has more than 200k celebrity images in total. We determined whether the person in a portrait image is wearing glasses or not. By training our algorithm on the training sets and tuning hyperparameters and regularisation adjustment we achieved an accuracy of approximately 95% on
the test images.

## Approach
### Data Extraction 
* We extracted the entire celeb dataset (images and labels) by reading the
and then for training the model, we chose different sizes of dataset.
Initially, just for testing purpose, we used 1000 data samples for training,
then we trained the model on 20K training samples, 70K samples and
finally on 2L dataset. We partitioned the dataset into training, validation
and test set.
* Below steps were followed:
  * Labels for eyeglasses were first extracted from the
list_attr_celeba.txt (column 16th). Converted the -1s label to 0s
depicting absence of glasses,1st depicted presence of glasses.
  * Read the images from img_align_celeba folder and converted the
RGB images into grayscale. This was done as we only have to
determine eyeglasses and it can be achieved even with a grayscale
image.
  * Resized the image to a smaller dimension (tried 28x28 and 32x32)
  * Saved the image names, image pixels data and labels corresponding
to each data in an npz file on the disk.
  * Standardized the image pixel values to be in a range of 0 to 1.
  * Flattened each image into 1D array
* Also, we tried a different variant of data extraction for 70K data samples.
  * In this, all the 202599 labels were read and the ones having
eyeglasses were extracted.
  * The total count of such images was 13193.
  * Then, we extracted 56807 non-eyeglass images and
combined both the sets, shuffled the data and created a new
dataset of size 70K.
  * This was done because the ratio of eyeglass images to
non-eyeglass images is too low in the current celeb dataset.
So, in order to train the model effectively, we needed a good
proportion of eyeglasses images in the dataset. This data
extraction method helped us to achieve it.
### Training and testing the CNN model
* Tried different variants of model as a part of hypertuning and selected the
optimum one which gives minimum validation error. We have attached
the dataset as the table in the hyperparameters tuning section.
* The model was first implemented, trained and tested using generic
Tensorflow libraries. In an effort to reduce the training time, tf.estimator
was used which improved the speed to a large extent.
* The model consists of 4 layers: Convolutional layer 1 (convolution with 32
features applied on 5x5 patch of image + 2D max pooling), Convolutional
layer 2 (convolution with 64 features applied on 5x5 patch of image + 2D
max pooling), fully connected layer with 1024 neurons and ReLU activation
function, logit layer with 2 units corresponding to 2 labels
* In fully connected layer, few neuron outputs are dropped out to prevent
overfitting. The no_drop_prob placeholder makes sure that dropout occurs
only during training and not during testing
* Tuned the model by varying different hyperparameters like dropout rate,
number of hidden layers, number of nodes in hidden layers, optimizer type
(GradientDescentOptimizer, AdamOptimizer) with learning rate set to 
0.001 and varying number of steps on input batches of size 100 and chose
that model which has the minimum cross-entropy error on validation set
* Then, ran the selected model on test celeb images to get test accuracy.
### Optimisations
* **Extracted feature values and labels from the data:** Since extracting the
features in original resolution severely clogged the system’s RAM, we
downsampled the images.
* **Reduced the resolution of the original images:** We worked with images resized
to dimension, 28x28
* **Reduced the size of training set:** Initially according to the the mentioned paper
we took 20000 training sets.
* **Performed Data Partition:** We partitioned the dataset into training, validation and test sets
after shuffling the data
* **Applied dropout or other regularization methods:** We experimented with
different dropout values, Details have been provided in the table in
Hyperparameters tuning section.
* **Trained model parameter:** We used SGD and AdamOptimizer with mini batches.
AdamOptimizer was found to be much faster and achieved the higher accuracy in
lower number of epochs
* **Tuned hyper-parameters:** We used the automate.py script to create the grid map
for different values of the dropout and hidden layers and hidden nodes.
* **Retrained the model using higher resolutions:** We used 32x32 image resolution.
The performance improved as we increased the resolution
* **Used bigger sizes of the training set:** We augmented the training set to 50K, 70K
and original dataset count.

## Results
<table>
  <thead><tr><th>Sr. No</th><th>Resolution</th><th>Dropout rate</th><th>Number of Convolutional Layers</th><th>Number of nodes</th><th>Train Accuracy</th><th>Validation Accuracy</th><th>Test Accuracy</th></tr></thead><tbody>
  <tr><td align="center" colspan="8"><b>Trained using SGD (Epoch 10000)</b></td></tr>
   <tr><td>1</td><td>28x28</td><td>0.3</td><td align="center" >2</td><td>32, 64</td><td>0.90134919</td><td>0.89892858</td><td>0.90742856</td></tr>
   <tr><td>2</td><td>28x28</td><td>0.3</td><td align="center" >2</td><td>64, 128</td><td>0.89954364</td><td>0.89910716</td><td>0.90721428</td></tr>
   <tr><td>3</td><td>28x28</td><td>0.3</td><td align="center" >2</td><td>128, 256</td><td>0.90037698</td><td>0.89928573</td><td>0.9069286</td></tr>
   <tr><td>4</td><td>28x28</td><td>0.4</td><td align="center" >2</td><td>32, 64</td><td>0.89515871</td><td>0.89285713</td><td>0.90135711</td></tr>
   <tr><td align="center" colspan="8"><b>Trained using AdamOptimiser (Epoch 1000)</b></td></tr>
   <tr><td>5</td><td>28x28</td><td>0.4</td><td align="center" >2</td><td>32, 64</td><td>0.94714284</td><td>0.94410712</td><td><span style="color:green"><b>0.94035715</b></span</td></tr>
   <tr><td>6</td><td>28x28</td><td>0.4</td><td align="center" >2</td><td>64, 128</td><td>0.93579364</td><td>0.93410712</td><td>0.93457144</td></tr>
   <tr><td>7</td><td>28x28</td><td>0.4</td><td align="center" >2</td><td>128, 256</td><td>0.92555553</td><td>0.92214286</td><td>0.92307144</td></tr>
   <tr><td>8</td><td>28x28</td><td>0.5</td><td align="center" >2</td><td>32, 64</td><td>0.93267858</td><td>0.93339288</td><td>0.93142855</td></tr>
   <tr><td>9</td><td>28x28</td><td>0.5</td><td align="center" >2</td><td>64, 128</td><td>0.94003969</td><td>0.93892854</td><td>0.93785715</td></tr>
   <tr><td>10</td><td>28x28</td><td>0.5</td><td align="center" >2</td><td>128, 256</td><td>0.93531746</td><td>0.93464285</td><td>0.93321431</td></tr>
</tbody></table>

## Documentation
***
Report and documentation can be found on this [Documentation](https://github.com/jayantsolanki/Proj-4-Introduction-to-Deep-Learning-IntroToML-574/blob/master/proj4.pdf) link

## Folder Tree
***
* [**Report**](https://github.com/jayantsolanki/Proj-4-Introduction-to-Deep-Learning-IntroToML-574/tree/master/Report) contains summary report detailing our implementation and results.
* [**code**](https://github.com/jayantsolanki/Proj-4-Introduction-to-Deep-Learning-IntroToML-574/tree/master/code)  contains the source code of our machine learning algorithm
* [**output**](https://github.com/jayantsolanki/Proj-4-Introduction-to-Deep-Learning-IntroToML-574/tree/master/proj4code) contains the console output of the analysis performed on the images

## Instructor
***
  * **Prof. Sargur N. Srihari**
  
## Teaching Assistants
***
  * **Jun Chu**
  * **Tianhang Zheng**
  * **Mengdi Huai**

## References
***
  * [Stackoverflow.com](Stackoverflow.com)
  * Python, Numpy and TensorFlow documentations
  * [https://cs231n.github.io/convolutional-networks/](https://cs231n.github.io/convolutional-networks/)
  * Liu, Ziwei; Luo, Ping; Wang, Xiaogang; Tang, Xiaoou. [“Deep Learning Face Attributes in the Wild”](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Liu_Deep_Learning_Face_ICCV_2015_paper.pdf)

## License
***
This project is open-sourced under [MIT License](http://opensource.org/licenses/MIT)
