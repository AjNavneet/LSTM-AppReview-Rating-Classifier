import re
from sklearn.preprocessing import LabelEncoder
import nltk
import time
import torch
import string
import numpy as np
from nltk.stem import PorterStemmer
import pandas as pd
from nltk.corpus import stopwords
from sklearn.utils import resample

class Preprocessing:
    """# Function to Clean the Text"""

    # Define a function to clean the textual data
    def clean_text(self, txt):
        """
        Clean and preprocess text data.
        :param txt: Input text
        :return: Cleaned text
        """
        txt = txt.lower()  # Convert text to lowercase
        txt = re.sub(r'\W', ' ', str(txt))  # Remove special characters (including apostrophes)
        txt = txt.translate(str.maketrans('', '', string.punctuation))  # Remove punctuations
        txt = ''.join([i for i in txt if not i.isdigit()]).strip()  # Remove digits
        txt = re.sub(r'\s+[a-zA-Z]\s+', ' ', txt)  # Remove single characters (e.g., 's')
        txt = re.sub(r'\s+', ' ', txt, flags=re.I)  # Substitute multiple spaces with a single space
        txt = re.sub(r"(http\S+|http)", "", txt)  # Remove links
        txt = ' '.join([PorterStemmer().stem(word=word) for word in txt.split(" ") if word not in stopwords.words('english')])  # Stem and remove stop words
        txt = ''.join([i for i in txt if not i.isdigit()]).strip()  # Remove digits
        return txt

    def sampling(self, data):
        """
        Perform upsampling of minority classes and downsampling of the majority class.
        :param data: Input data with class labels
        :return: Balanced data
        """
        # Separate data into majority and minority classes
        df_majority = data[data['score'] == 5]  # Majority class with score 5
        df_minority1 = data[data['score'] == 2]  # Minority class with score 2
        df_minority2 = data[data['score'] == 3]  # Minority class with score 3
        df_minority3 = data[data['score'] == 1]  # Minority class with score 1
        df_minority4 = data[data['score'] == 4]  # Minority class with score 4

        """# Upasampling the Monority class and Downsampling the Majority Class"""
        # Downsample the majority class (score 5)
        df_majority_downsampled = resample(df_majority, replace=False, n_samples=600)

        # Upsample the minority classes (score 2, 3, 1, 4)
        df_minority_upsampled = resample(df_minority1, replace=True, n_samples=200)
        df_minority_upsampled1 = resample(df_minority2, replace=True, n_samples=300)
        df_minority_upsampled2 = resample(df_minority3, replace=True, n_samples=225)
        df_minority_upsampled3 = resample(df_minority4, replace=True, n_samples=250)

        # Combine the minority classes with the downsampled majority class
        data1 = pd.concat([df_majority_downsampled,
                           df_minority_upsampled,
                           df_minority_upsampled1,
                           df_minority_upsampled2,
                           df_minority_upsampled3])

        return data1

    def encoder(self, data1):
        """
        Encode class labels using a label encoder.
        :param data1: Data with class labels
        :return: Label encoder and encoded labels
        """
        le = LabelEncoder()
        Y = le.fit_transform(data1['score'])
        print(Y.shape)
        print(le.classes_)
        return le, Y
