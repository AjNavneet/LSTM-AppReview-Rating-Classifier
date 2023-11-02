# Define parameters
max_features = 2000
batch_size = 50
vocab_size = max_features

# Import necessary libraries
from sklearn.model_selection import train_test_split
import torch

class Create:

    def create_dataset(self, X, Y):
        """
        Split the dataset into training and testing data.
        :param X: Input data (features)
        :param Y: Output data (labels)
        :return: Training and testing data splits
        """
        # Train and Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
        print(X_train.shape, Y_train.shape)
        print(X_test.shape, Y_test.shape)
        return X_train, X_test, Y_train, Y_test

    def data_loader(self, X_train, X_test, Y_train, Y_test):
        """
        Convert the data into PyTorch tensors and create data loaders.
        :param X_train: Training data features
        :param X_test: Testing data features
        :param Y_train: Training data labels
        :param Y_test: Testing data labels
        :return: PyTorch tensors and data loaders
        """
        # Convert data to PyTorch tensors
        x_train = torch.tensor(X_train, dtype=torch.long)
        y_train = torch.tensor(Y_train, dtype=torch.long)
        x_cv = torch.tensor(X_test, dtype=torch.long)
        y_cv = torch.tensor(Y_test, dtype=torch.long)

        # Convert dataset to a PyTorch Dataset
        train = torch.utils.data.TensorDataset(x_train, y_train)
        valid = torch.utils.data.TensorDataset(x_cv, y_cv)

        # Initialize the DataLoaders
        train_dl = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        val_dl = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
        
        return x_cv, y_cv, train_dl, val_dl
