############################################################
#### Project Title:      HandWrittenDigitsRecognitionNN ####
#### Author:             Behnaz Mohammad Hasani Zadeh   ####
#### Date:               6/2/2024                       ####
#### Description:        This project recognizes hand   ####
####                     written digits from MNIST      ####
####                     database using Neural Network. ####  
############################################################

## Importimg libraries needed ##
from sklearn.neural_network import MLPClassifier                       #To use the mlp model
from sklearn.datasets import load_digits                               #To use the database
from sklearn.model_selection import train_test_split                   #To do the test train split
import matplotlib.pyplot as plt                                        #To show the image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay   # To make the confusion matrix                 


## Main ##

############# Load and return the digits dataset ###############
# Each datapoint is a 8x8 image of a digit.                    #
x, y = load_digits(return_X_y = True) 


######################## First Try #############################
# Using whole dataset to train and test                        #
# Number of hidden layers:  1                                  #
# Number of neurons in hidden layer:                           #
#                             65 (equal to number of features) #
# Activation function for the hidden layer: sigmoid            #
# Learning Rate:            0.01                               #
# mlp = MLPClassifier(hidden_layer_sizes = (65,), activation = 'logistic', learning_rate_init = 0.01)

# # Training model #
# mlp.fit(x, y)

# # Mean accuracy  #
# print("Score:", mlp.score(x, y))

# # Confusion Matrix #
# y_pred = mlp.predict(x) 
# cm = confusion_matrix(y, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = mlp.classes_)
# disp.plot(cmap = plt.cm.Blues)
# plt.show()


######################## Second Try ############################
# Using 50% of dataset to train and 50% to test.               #
# Number of hidden layers:           1                         #
# Number of neurons in hidden layer:                           #
#                             65 (equal to number of features) #
# Activation function for the hidden layer: sigmoid            #
# Learning Rate:                     0.01                      #
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.5, test_size = 0.5, random_state = 2)

# mlp = MLPClassifier(hidden_layer_sizes = (65,), activation = 'logistic', learning_rate_init = 0.01)

# # Training model #
# mlp.fit(x_train, y_train)

# # Mean accuracy  #
# print(mlp.score(x_test, y_test))

# # Confusion Matrix #
# y_pred = mlp.predict(x_test) 
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = mlp.classes_)
# disp.plot(cmap = plt.cm.Blues)
# plt.show()

# Finding the first incorrecr predicted input #
# incorrect = x_test[y_pred != y_test]
# incorrect_true = y_test[y_pred != y_test]
# incorrect_pred = y_pred[y_test != y_pred]
# j = 0
# plt.matshow(incorrect[j].reshape(8, 8), cmap=plt.cm.gray)
# plt.xticks(())
# plt.yticks(())
# plt.show()
# print("true value:", incorrect_true[j])
# print("predicted value:", incorrect_pred[j])


######################## Third Try #############################
# Using 20% of dataset to train and 80% to test.               #
# Number of hidden layers:           1                         #
# Number of neurons in hidden layer:                           #
#                             65 (equal to number of features) #
# Activation function for the hidden layer: sigmoid            #
# Learning Rate:                     0.01                      #
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.2, test_size = 0.8, random_state = 2)

mlp = MLPClassifier(hidden_layer_sizes = (65,), activation = 'logistic', learning_rate_init = 0.01)

# Training model #
mlp.fit(x_train, y_train)

# Mean accuracy  #
print(mlp.score(x_test, y_test))

# Confusion Matrix #
y_pred = mlp.predict(x_test) 
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = mlp.classes_)
disp.plot(cmap = plt.cm.Blues)
plt.show()

# Finding the first incorrecr predicted input #
incorrect = x_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_test != y_pred]
j = 2
plt.matshow(incorrect[j].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print("true value:", incorrect_true[j])
print("predicted value:", incorrect_pred[j])

