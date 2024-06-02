############################################################
#### Project Title:      HandWrittenDigitsRecognitionNN ####
#### Author:             Behnaz Mohammad Hasani Zadeh   ####
#### Date:               6/2/2024                       ####
#### Description:        This project recognizes hand   ####
####                     written digits from MNIST      ####
####                     database using Neural Network. ####  
############################################################

## Importimg libraries needed ##
from sklearn.neural_network import MLPClassifier      #To use the mlp model
from sklearn.datasets import load_digits              #To use the database
from sklearn.model_selection import train_test_split  #To do the test train split
import matplotlib.pyplot as plt                       #To show the image


## Main ##

############# Load and return the digits dataset ###############
# Each datapoint is a 8x8 image of a digit.                    #
x, y = load_digits(return_X_y = True) 


######################## First Try #############################
# Using whole dataset to train and test                        #
# Number of hidden layers:           1                         #
# Number of neurons in hidden layer:                           #
#                             64 (equal to number of features) #
# Activation function for the hidden layer: sigmoid            #
# Learning Rate:                     0.05                       #
mlp = MLPClassifier(hidden_layer_sizes = (64,), activation = 'logistic', learning_rate_init = 0.04)

# Training model #
mlp.fit(x, y)

# Mean accuracy  #
print(mlp.score(x, y))