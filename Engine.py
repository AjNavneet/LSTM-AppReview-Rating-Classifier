# Import necessary libraries and dependencies
import re
import tensorflow as tf
import numpy as np
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

# Import additional components
from tensorflow import keras
import torch.nn.functional as F
# (optional) Import a library for visualization, e.g., matplotlib.pyplot as plt

# Import NLTK's Porter Stemmer for text processing
from nltk.stem import PorterStemmer

# Import scikit-learn for train-test split
from sklearn.model_selection import train_test_split

# Import Tokenizer and padding functions from TensorFlow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import custom components from the MLPipeline package
from MLPipeline.Load_Data import Load_Data
from MLPipeline.Preprocessing import Preprocessing
from MLPipeline.Tokenisation import Tokenisation
from MLPipeline.Create import Create
from MLPipeline.Lstm import LSTM
from MLPipeline.Train_Test import Train_Test

# Define some constants
max_features = 2000
batch_size = 50
vocab_size = max_features

# Load the data using the Load_Data class or function and print it
data = Load_Data().load_data()
print(data)

# Initialize a Preprocessing instance for text cleaning
preprocess = Preprocessing()

"""# Applying the Function to the Dataset"""

# Apply the 'clean_text' function to every text in the 'content' column of the dataset
data['content'] = data['content'].apply(preprocess.clean_text)

# We can see that class "5" is dominating in the dataset. Thus, we need to balance the dataset.

"""# Balancing the Dataset"""

# Apply sampling to balance the dataset (specific details not provided)

""" # Tokenisation """
# Generate a tokenization dictionary and tokenize the data using the Tokenisation class
word_index, X = Tokenisation().generate_token(data1)

"""# Label Encoding """
# Perform label encoding on the dataset using the 'encoder' function from preprocessing
le, Y = preprocess.encoder(data1)

# Create a 'Create' instance for creating datasets and data loaders
creating = Create()

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = creating.create_dataset(X, Y)

# Create data loaders for training and validation data
x_cv, y_cv, train_dl, val_dl = creating.data_loader(X_train, X_test, Y_train, Y_test)

"""# Defining the Model"""
# Define the LSTM model with specified parameters
model = LSTM(vocab_size, 128, 64)
print(model)

# Train and test the model using the 'train_test' function from the Train_Test class
Train_Test().train_test(10, model, train_dl, x_cv, val_dl, Y_test)
