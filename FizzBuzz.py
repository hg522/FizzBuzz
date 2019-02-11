# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 02:12:39 2018

@author: Himanshu Garg
UBPerson No: 50292195
"""

#logic based fizzbuss implementation function for Software 1.0
import pandas as pd
import sklearn.metrics as skm

def fizzbuzz(n):

    # Logic Explanation
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return 'Other'


#Create Training and Testing Datasets in CSV Format
def createInputCSV(start,end,filename):

    # Why list in Python?
    """We use a list here since it is ordered and changeable, and we can easily
    access items using indexes. Also, it is easier to loop through the list
    and add (using append) and remove items from it.
    """
    inputData   = []
    outputData  = []

    # Why do we need training Data?
    """We need training data to train the model or classifier using the machine 
    learning algorithm to learn the pattern of occurence of the fizz, buzz, fizzbuzz
    and Other, so that when the trained model is applied to any test data, then we 
    are able to predict or achieve the correct result.
    """
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))

    # Why Dataframe?
    """A Dataframe in python is defined as 2-D labeled data structure with columns
    of different types. Since we want to create a structure with both inputs and labels,
    dataframe is the best fit.
    """
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData

    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)

    print(filename, "Created!")



#Processing Input and Label Data
def processData(dataset):

    # Why do we have to process?
    """Since we have a categorical data with labels we need to convert it into
    numerical representation for the machine learning algorithm to use. In Keras
    The input is in the form of numpy arrays. Hence we process to convert to
    binary form for the keras model to understand.
    """
    data   = dataset['input'].values
    labels = dataset['label'].values

    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)

    return processedData, processedLabel


def encodeData(data):

    processedData = []

    for dataInstance in data:

        # Why do we have number 10?
        """Since our domain space is from 1 to 1000 including the test cases,
        we need 10 bits to convert to binary form, hence to 
        represent for eg 1000, we need 10 bits.
        """
        processedData.append([dataInstance >> d & 1 for d in range(10)])

    return np.array(processedData)


from keras.utils import np_utils

def encodeLabel(labels):

    processedLabel = []

    for labelInstance in labels:
        if(labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])
    #here we use 4 since there are four categories/class of data
    return np_utils.to_categorical(np.array(processedLabel),4)



#Model Definition
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard

import numpy as np

input_size = 10
drop_out_1 = 0.1
#drop_out_2 = 0.5
first_dense_layer_nodes  = 1024
#second_dense_layer_nodes = 512
second_dense_layer_nodes = 4

def get_model():

    # Why do we need a model?
    """We need a model to store the output or the wieghts of the machine learning
    algorithm that would be used to determine the output of the test data.It
    basically means the classifier that we create and train to finally apply
    to the test data.
    """
    # Why use Dense layer and then activation?
    """Dense layer declares the number of neurons, weight and biases to perform
    the linear transformation on the data. It is easy to solve and are generally
    present in most neural networks. Activation function is extremely important as 
    it decides whether a neuron is to be activated or not. It is required to perform
    non-linear transformation on the data so that it can learn and solve more
    complex tasks i.e. it is able to better fit the training data. This output is 
    passed to the next layer.
    Using activation function makes the back-propagation process possible as the
    along with errors, gradients are also sent to update the weights and biases
    appropriately.
    """
    # Why use sequential model with layers?
    """Sequential model is simply a stack of layers and can be created
    by passing layer instances to the constructor. It is simple to use and since
    we have one source of inputs and not multiple, this is a good fit.
    """
    model = Sequential()
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))

    # Why dropout?
    """We need dropout to make sure that overfitting doesn't occur. i.e.
    some fraction of the nodes are dropped randomply so that machine learning 
    algorithm while training with a large number of epochs doesn't get biased,
    and is able to learn correctly.
    """
    model.add(Dropout(drop_out_1))
    #model.add(Dense(second_dense_layer_nodes))
    #model.add(Activation('sigmoid'))
    #model.add(Dropout(drop_out_2))
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))

    # Why Softmax?
    """Softmax is used when we have a classification problem. It is generally used
    in the output layer of the classifier where we are trying to determine the 
    probabilities to determine the class of the inputs. We have used it here because
    it can handle multiple classes like in our case, and compresses the outputs between
    [0,1] and also divides by the output doing the normalisation, helping in the analysis.
    """
    model.summary()

    # Why use categorical_crossentropy?
    """categorical_crossentropy loss function is used when we have multi-class
    classification problem. In our case we have 4 classes. Our aim here is to 
    minimize this loss function to improve the accuracy.
    """
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


#Creating Training and Testing Datafiles
# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')


#Creating Model
model = get_model()

#Running the Model
validation_data_split = 0.2
num_epochs = 10000
model_batch_size = 128
tb_batch_size = 32
early_patience = 100

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

# Read Dataset
dataset = pd.read_csv('training.csv')

# Process Dataset
processedData, processedLabel = processData(dataset)
history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )


#print(history.history.keys())


import matplotlib.pyplot as plt
#import pylab as pl
#Training and Validation Graphs
#%matplotlib inline
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))
plt.show()

#Testing Accuracy [Software 2.0]
def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"


wrong   = 0
right   = 0
fright = 0
fwrong = 0
bright = 0
bwrong = 0
fbright = 0
fbwrong = 0
oright = 0
owrong = 0

def updateClassAccFlag(label,status):
    global fbright
    global fbwrong
    global fright
    global fwrong
    global bright
    global bwrong
    global oright
    global owrong

    if label == "FizzBuzz":
        if status == True:
            fbright = fbright + 1
        else:
            fbwrong = fbwrong + 1
    elif label == "Fizz":
        if status == True:
            fright = fright + 1
        else:
            fwrong = fwrong + 1
    elif label == "Buzz":
        if status == True:
            bright = bright + 1
        else:
            bwrong = bwrong + 1
    else:
        if status == True:
            oright = oright + 1
        else:
            owrong = owrong + 1

testData = pd.read_csv('testing.csv')

processedTestData  = encodeData(testData['input'].values)
processedTestLabel = encodeLabel(testData['label'].values)
predictedTestLabel = []

for i,j in zip(processedTestData,processedTestLabel):
    y = model.predict(np.array(i).reshape(-1,10))
    predictedTestLabel.append(decodeLabel(y.argmax()))

    if j.argmax() == y.argmax():
        right = right + 1
        updateClassAccFlag(decodeLabel(y.argmax()),True)
    else:
        wrong = wrong + 1
        updateClassAccFlag(decodeLabel(y.argmax()),False)

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))
if(fright == 0):
    print("\nFizz Accuracy: " + str(0))
else:
    print("Fizz Accuracy: " + str(fright/(fright+fwrong)*100))
if(bright == 0):
    print("Buzz Accuracy: " + str(0))
else:
    print("Buzz Accuracy: " + str(bright/(bright+bwrong)*100))
if(fbright == 0):
    print("FizzBuzz Accuracy: " + str(0))
else:
    print("FizzBuzz Accuracy: " + str(fbright/(fbright+fbwrong)*100))
if(oright == 0):
    print("Other Accuracy: " + str(0))
else:
    print("Other Accuracy: " + str(oright/(oright+owrong)*100))

print("\nConfusion Matrix: ")

confmatrix = skm.confusion_matrix(testData['label'].tolist(),predictedTestLabel,labels=["Fizz","Buzz","FizzBuzz","Other"])
print(confmatrix)

# Please input your UBID and personNumber
testDataInput = testData['input'].tolist()
testDataLabel = testData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "hgarg")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50292195")

predictedTestLabel.insert(0, "")
predictedTestLabel.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabel

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')

