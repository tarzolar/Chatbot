# -*- coding: utf-8 -*-
"""
@author: Tomas Arzola Röber
Implementation of a Chatbot using an artificial neural network and PyTorch

"""
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
nltk.download('punkt')

df = pd.read_csv(r"C:\Users\HP\Downloads\Chatbot\ChatbotTraining.csv")
df = df.sample(frac=1).reset_index(drop=True)
label_encoder = LabelEncoder()
df['tag'] = label_encoder.fit_transform(df['tag'])
y = df['tag']
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

words = []
for pattern in X:
    words.extend(tokenize(pattern))
    
for i in range(len(X)):
    X.iloc[i] = bag_of_words(X.iloc[i], words).tolist()


# Default Hyperparameter
epochs = 10
learning_rate = 0.001
input_nodes = len(words)
hidden_nodes = 100
output_nodes = len(y.unique())


class NN(nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_nodes, hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(hidden_nodes, output_nodes),
            )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits
    
    
class Data(Dataset):
    def __init__(self):
        self.x = torch.tensor(X)
        self.y = torch.tensor(y)
        self.n_samples = len(df)
        
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    
    def __len__(self):
        return self.n_samples
    

# Test Function
def test(model, test_loader):
    correct = []
    for idx, (data, label) in enumerate(test_loader):
        output = model(data.float())
        output = output.argmax()
        if label == output:
            correct.append(1)
        else:
            correct.append(0)
    return sum(correct)/test_size


# Training
torch.manual_seed(123)
dataset = Data()
dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=1)

learning_rates = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  
epochs_ = [1,2,3,5,10,20]
rates_and_epochs = []
performances = []

train_size = int(0.8*len(df))
test_size = len(df) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(Data(), [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, shuffle=True)
torch.manual_seed(123)


for rate in learning_rates:
    model = NN(input_nodes, hidden_nodes, output_nodes)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=rate)
    for epoch in epochs_:
        print(f"Training for learing rate {rate} and epochs number {epoch}")
        rates_and_epochs.append((rate, epoch))
        for e in range(epoch):
            for idx, (data, labels) in enumerate(train_loader):
                labels = labels.type(torch.LongTensor)
                outputs = model(data)
                loss = loss_function(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        performances.append(test(model, test_loader))

optimal_rate = rates_and_epochs[np.argmax(performances)][0]
optimal_epochs = rates_and_epochs[np.argmax(performances)][1]
print(f"Optimal learning rate {optimal_rate} and optimal epochs number {optimal_epochs}")
   
model = NN(input_nodes, hidden_nodes, output_nodes)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=optimal_rate) 

print()
for epoch in range(optimal_epochs):
    print(f"Training Epoch {epoch}")
    for idx, (data, labels) in enumerate(train_loader):
        labels = labels.type(torch.LongTensor)
        
        outputs = model(data)
        loss = loss_function(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(f"Performance: {test(model, test_loader)}")


model = NN(input_nodes, hidden_nodes, output_nodes)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=optimal_rate) 

performances = np.array(performances).reshape(len(learning_rates), len(epochs_))
plt.imshow(performances, cmap='coolwarm')
plt.xlabel('Number of Epochs')
plt.ylabel('Learning Rates')
plt.xticks(np.arange(len(epochs_)), epochs_)
plt.yticks(np.arange(len(learning_rates)), learning_rates)
plt.colorbar(label='Performance')
plt.title('Performances Heatmap')
plt.show()

print()
for epoch in range(optimal_epochs):
    print(f"Training Epoch {epoch}")
    for idx, (data, labels) in enumerate(dataloader):
        labels = labels.type(torch.LongTensor)
        
        outputs = model(data)
        loss = loss_function(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Chatbot
while(True):
      print("User:")
      user = torch.tensor([bag_of_words(input(), words)])
      df_r = df[df['tag'] == int(torch.argmax(model(user)))]
      response = df_r['responses'].sample().values[0]
      print("Chatbot:")
      for leter in response:
          print(leter, end='', flush=True)  
          time.sleep(0.05) 
      print()
