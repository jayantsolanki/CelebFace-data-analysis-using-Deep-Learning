# Proj-4-Introduction-to-Deep-Learning-IntroToML-574
Celebrity Image classification: Identifying celebrities who are wearing spectacles or not
## Explanations goes here
### Motivation
### Approach
### Results

![University at Buffalo](https://commons.wikimedia.org/wiki/File:University_at_Buffalo_logo.png)
***
# CelebFace data analysis using a Convolution Neural Network

## Introduction
***
In this project we perfomed machine learning analysis using CNN the Celeb dataset which has more than 200k celebrity images in total. We determined whether the person in a portrait image is wearing glasses or not. By training our algorithm on the training sets and tuning hyperparameters and regularisation adjustment we achieved an accuracy of approximately 95% on
the test images.

## Results
Sr.No | Resolution | Dropout-rate | Number of Convolutional Layers | Number of nodes | Train Accuracy | Validation Accuracy Test Accuracy
Trained using SGD (Epoch 10000)
1 28x28 0.3 2 32, 64 0.90134919 0.89892858 0.90742856
2 28x28 0.3 2 64, 128 0.89954364 0.89910716 0.90721428
3 28x28 0.3 2 128, 256 0.90037698 0.89928573 0.9069286
4 28x28 0.4 2 32, 64 0.89515871 0.89285713 0.90135711
Trained using AdamOptimiser (Epoch 1000)
5 28x28 0.4 2 32, 64 0.94714284 0.94410712 0.94035715
6 28x28 0.4 2 64, 128 0.93579364 0.93410712 0.93457144
4  
Project 4, CSE-574 Jayant Solanki, Swati Nair
7 28x28 0.4 2 128, 256 0.92555553 0.92214286 0.92307144
8 28x28 0.5 2 32, 64 0.93267858 0.93339288 0.93142855
9 28x28 0.5 2 64, 128 0.94003969 0.93892854 0.93785715
10 28x28 0.5 2 128, 256 0.93531746 0.93464285 0.93321431

## Documentation
***
Report and documentation can be found on this [Documentation](https://github.com/jayantsolanki/Proj-4-Introduction-to-Deep-Learning-IntroToML-574/blob/master/proj4.pdf) link

## Folder Tree
***
* **Report** contains summary report detailing our implementation and results.
* **code**  contains the source code of our machine learning algorithm
* **output** contains the console output of the analysis performed on the images

## Contributors
***
  * [Jayant Solanki](https://github.com/jayantsolanki)
  * [Swati S. Nair](https://github.com/swaitshr)
  
## Instructor
***
  * **Prof. Sargur N. Srihari**
  
## Teaching Assistants
***
  * **Jun Chu**
  * **Tianhang Zheng**
  * **Mengdi Huai**

## License
***
This project is open-sourced under [MIT License](http://opensource.org/licenses/MIT)
