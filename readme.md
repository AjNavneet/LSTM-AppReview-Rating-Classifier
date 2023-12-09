# LSTM Text Classification using PyTorch for App Reviews

## Overview

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture used in deep learning. LSTMs are specifically designed to handle long-term dependencies in data, making them well-suited for tasks involving text data, speech, and time series. In this project, we build an LSTM model to classify app reviews on a scale of 1 to 5 based on user feedback using PyTorch.

---

### Aim

To build text classifier to classify app reviews on a scale of 1 to 5 using LSTM.

---

### Data Description

The dataset consists of app reviews and corresponding ratings. The "score" column contains ratings in the range of 1 to 5, and the "content" column contains the review text.

---


### Tech Stack

- Language: `Python`
- Libraries: `pandas`, `TensorFlow`, `matplotlib`, `scikit-learn`, `NLTK`, `NumPy`, `PyTorch`

---

## Approach

### Data Preprocessing

1. Lowercasing text, removing punctuation, and eliminating links.
2. Balancing classes.
3. Tokenizing the text.
4. Scaling the data.

### Model

- Training an LSTM model in PyTorch.

### Model Evaluation

- Evaluating the model on test data.

---

## Modular Code Overview

1. **Input**: Contains the data used for analysis, including:
   - [List of data files]

1. **ML_Pipeline**: This folder contains functions distributed across multiple Python files, each appropriately named for its functionality. These functions are called from the `Engine.py` file.

2. **Notebook**: Contains the Jupyter Notebook file of the project.

3. **Engine.py**: The main script that orchestrates the different parts of the project by calling functions from the ML Pipeline.

4. **Readme.md**: Instructions for running the code and additional information about the project.

5. **requirements.txt**: Lists all the required libraries and their versions for easy installation using `pip`.

---
