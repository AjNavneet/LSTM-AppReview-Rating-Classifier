# Pytorch

- PyTorch is an open source machine learning library for Python and is completely based on Torch. 
- It is primarily used for applications such as natural language processing
- PyTorch redesigns and implements Torch in Python while sharing the same core C libraries for the backend code.
- PyTorch developers tuned this back-end code to run Python efficiently. They also kept the GPU based hardware acceleration as well as the extensibility features that made Lua-based Torch.

## Recurrent Neural Net
- Recurrent Neural Network(RNN) are a type of Neural Network where the output from previous step are fed as input to the current step.
  In traditional neural networks, all the inputs and outputs are independent of each other, but in cases like when it is required to predict the next word of a sentence, the previous words are required and hence there is a need to remember the previous words. 
  Thus RNN came into existence, which solved this issue with the help of a Hidden Layer. 
- The main and most important feature of RNN is Hidden state, which remembers some information about a sequence.
  RNN have a “memory” which remembers all information about what has been calculated. It uses the same parameters for each input as it performs the same task on all the inputs or hidden layers to produce the output. This reduces the complexity of parameters, unlike other neural networks.


## Architecture

- The basic difference between the architectures of RNNs and LSTMs is that the hidden layer of LSTM is a gated unit or gated cell.

- It consists of four layers that interact with one another in a way to produce the output of that cell along with the cell state. These two things are then passed onto the next hidden layer. 

- Unlike RNNs which have got the only single neural net layer of tanh, LSTMs comprises of three logistic sigmoid gates and one tanh layer. Gates have been introduced in order to limit the information that is passed through the cell.

- They determine which part of the information will be needed by the next cell and which part is to be discarded. The output is usually in the range of 0-1 where ‘0’ means ‘reject all’ and ‘1’ means ‘include all’.  



## Code Description

    File Name :Load_Data.py
    File Description : Load the Dataset
    
    File Name :Preprocessing.py
    File Description : Class to preprocess the dataset
    
    File Name :Tokenisation.py
    File Description : Class to Tokenise the dataset

    File Name : Create.py
    File Description : Code to load and transform the dataset to torch dataset.

    File Name : LSTM.py
    File Description : Class of LSTM structure

    File Name : Train_test.py
    File Description : Code to train and evaluate the pytorch model

    File Name : Engine_Final.py
    File Description : Main class for starting the model training lifecycle



## Steps to Run

There are two ways to execute the end to end flow.

- Modular Code
- IPython

### Modular code

- Create virtualenv
- Install requirements `pip install -r requirements.txt`
- Run Code `python Engine_Final.py`
- Check output for all the visualization

### IPython Google Colab

Follow the instructions in the notebook `text_classif_LSTM.ipynb`

