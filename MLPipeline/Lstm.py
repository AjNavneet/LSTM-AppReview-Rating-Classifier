import torch
import torch.nn as nn

class LSTM(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        Initialize the LSTM model.
        :param vocab_size: Size of the vocabulary (number of unique words)
        :param embedding_dim: Dimension of word embeddings
        :param hidden_dim: Dimension of the hidden state in the LSTM
        """
        super().__init()
        
        # Define model components
        self.hidden_dim = hidden_dim
        
        # Embedding layer for word representations
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm1 = nn.LSTM(hidden_dim, 64, batch_first=True)
        
        # Linear layer for classification
        self.linear = nn.Linear(64, 5)

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: Input data (word indices)
        :return: Model predictions
        """
        # Embed the input data
        x = self.embeddings(x)
        
        # Pass through the first LSTM layer
        out_pack, (ht, ct) = self.lstm(x)
        
        # Pass through the second LSTM layer
        out_pack1, (ht, ct) = self.lstm1(out_pack)
        
        # Apply a linear layer for classification and get predictions
        out = self.linear(ht[-1])
        
        return out
