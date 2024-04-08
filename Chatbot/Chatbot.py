# -*- coding: utf-8 -*-
"""
Implementation of a Chatbot using an artificial neural network

@author: Tomas Arzola Röber
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import PorterStemmer
nltk.download('punkt')

# Importing the data set
df = pd.read_csv(r"ChatbotTraining.csv")
label_encoder = LabelEncoder()
df['tag'] = label_encoder.fit_transform(df['tag'])
y = df['tag']
y = label_encoder.fit_transform(y)
y = pd.DataFrame(y)
X = df['patterns']
responses = df['responses']

# Tokenizes a sentence
def tokenize(sentence):
    """
    Tokenizes a sentence into individual words.
    
    Args:
    sentence (str): The input sentence to tokenize.
    
    Returns:
    list: A list of tokenized words.
    """
    return nltk.word_tokenize(sentence)

# Returns the stem of the word 
def stem(word):
    """
    Returns the stem of a word.
    
    Args:
    word (str): The input word to stem.
    
    Returns:
    str: The stemmed word.
    """
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())

# Returns a bag of words of a sentence
def bag_of_words(sentence, words):
    """
    Generates a bag of words representation of a sentence.
    
    Args:
    sentence (str): The input sentence.
    words (list): List of stemmed words in the dataset.
    
    Returns:
    numpy.ndarray: Bag of words representation of the input sentence.
    """
    sentence = tokenize(sentence)
    sentence_words = [stem(word) for word in sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for i in range(len(words)):
        if words[i] in sentence_words:
            bag[i] = 1
    return bag
 

# Saves the words of the training data set
words = []
for pattern in X:
    words.extend(tokenize(pattern))
    
# Default values of the hyperparameters
epochs = 10
learning_rate = 0.3
input_nodes = len(words)
hidden_nodes = 512
output_nodes = len(y[0].unique())

# Weights matrix with random values
w_ih = np.random.uniform(-0.5, 0.5, size=(hidden_nodes, input_nodes))
w_ho = np.random.uniform(-0.5, 0.5, size=(output_nodes, hidden_nodes))

w_ihs = w_ih
w_hos = w_ho

# Feedforward function with linear algebra
def feedforward(input_vector, w_ih, w_ho):
    """
    Performs the feedforward operation of the neural network.
    
    Args:
    input_vector (numpy.ndarray): Input vector to the neural network.
    w_ih (numpy.ndarray): Weights matrix between input and hidden layer.
    w_ho (numpy.ndarray): Weights matrix between hidden and output layer.
    
    Returns:
    output_final (numpy.ndarray): Output vector of the neural network.
    output_hidden (numpy.ndarray): Output vector of the hidden layer.
    """
    input_hidden = np.dot(w_ih, input_vector)
    output_hidden = expit(input_hidden)
    input_final = np.dot(w_ho, output_hidden)
    output_final = expit(input_final)
    return output_final, output_hidden

# Calculates the error and corrects the wights with backpropagation
def train(input_vector, target, w_ih, w_ho, learning_rate):
   """
   Performs one training step using backpropagation.
    
   Args:
   input_vector (numpy.ndarray): Input vector to the neural network.
   target (numpy.ndarray): Target output vector.
   w_ih (numpy.ndarray): Weights matrix between input and hidden layer.
   w_ho (numpy.ndarray): Weights matrix between hidden and output layer.
   learning_rate (float): Learning rate for updating weights.
   """
 
   output_final, output_hidden = feedforward(input_vector, w_ih, w_ho)
   target = np.array(target, ndmin=2).T
   input_vector = np.array(input_vector, ndmin=2).T
   output_hidden = np.array(output_hidden, ndmin=2).T
   output_final = np.array(output_final, ndmin=2).T
   final_errors = target - output_final
   hidden_errors = np.dot(w_ho.T, final_errors)

   
   w_ho += learning_rate*(np.dot((final_errors*output_final*(1-output_final)), output_hidden.T))
   w_ih += learning_rate*(np.dot((hidden_errors*output_hidden*(1-output_hidden)), input_vector.T))
   

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)

# Test function to calculate the performance of the neural network
def test(X_test, y_test, w_ih, w_ho):
    """
    Tests the performance of the neural network.
    
    Args:
    X_test (pandas.Series): Testing input data.
    y_test (pandas.Series): Testing target data.
    w_ih (numpy.ndarray): Weights matrix between input and hidden layer.
    w_ho (numpy.ndarray): Weights matrix between hidden and output layer.
    
    Returns:
    performance (float): Performance of the neural network.
    """
    correct = []
    for i in range(len(X_test)):
        target = int(y_test.iloc[i])
        output = feedforward(bag_of_words(X_test.iloc[i], words), w_ih, w_ho)[0]

        if np.argmax(output) == target:
            correct.append(1)
        else:
            correct.append(0)
    performance = sum(correct)/len(X_test)
    return performance
 
# Learning rate optimization   
learning_rates = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]        
performances_for_rates = []

for rate in learning_rates:
    for i in range(len(X_train)):
        target_vector = np.zeros(output_nodes)
        target_vector[int(y_train.iloc[i])] = 0.99
        train(bag_of_words(X_train.iloc[i], words), target_vector, w_ih, w_ho, rate)
            
    performances_for_rates.append(test(X_test, y_test, w_ih, w_ho))
    w_ih = w_ihs
    w_iho = w_hos
    print(f"Training rate "+str(rate))
        
plt.plot(learning_rates, performances_for_rates)
plt.xlabel('Learning rates')
plt.ylabel('Performance')
plt.title('Performance vs. Learning rates')
plt.show()        
print()
print(f"Optimal learning rate:"+str(learning_rates[np.argmax(performances_for_rates)]))
print()

# Epochs number optimization
epochs_ = [1,2,3,5,10,20]
performances_for_epochs = []

for epoch in epochs_:
    for e in range(epoch):
        for i in range(len(X_train)):
            target_vector = np.zeros(output_nodes)
            target_vector[int(y_train.iloc[i])] = 0.99
            train(bag_of_words(X_train.iloc[i], words), target_vector, w_ih, w_ho, 0.3)

    performances_for_epochs.append(test(X_test, y_test, w_ih, w_ho)) 
    w_ih = w_ihs
    w_ho = w_hos
    print("Training epoch: "+str(epoch))
        
plt.plot(epochs_, performances_for_epochs)
plt.xlabel('Number of epochs')
plt.ylabel('Performance')
plt.title('Performance vs. Number of epochs')
plt.show()        
print()
print(f"Optimal number of epochs:"+str(epochs_[np.argmax(performances_for_epochs)]))       
print()

# Training with the optimal learning rate and number of epochs
print("Starting Training")
for e in range(epochs_[np.argmax(performances_for_epochs)]):
    for i in range(len(X)):
        target_vector = np.zeros(output_nodes)
        target_vector[int(y.iloc[i])] = 0.99
        train(bag_of_words(X.iloc[i], words), target_vector, w_ih, w_ho, learning_rates[np.argmax(performances_for_rates)])
print("Training Finished")
print()
model_performance = test(X_test, y_test, w_ih, w_ho)
print("Model performance: "+str(model_performance))
print()

# Chatbot
while(True):
     print("User:")
     user = input()
     df_r = df[df['tag'] == np.argmax(feedforward(bag_of_words(user,words),w_ih, w_ho)[0])]
     response = df_r['responses'].sample().values[0]
     print("Chatbot:")
     for letra in response:
         print(letra, end='', flush=True)  
         time.sleep(0.05) 
     print()
