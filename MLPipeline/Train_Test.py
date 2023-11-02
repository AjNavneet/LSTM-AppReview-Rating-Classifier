import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class Train_Test:

    def train_test(self, n_epochs, model, train_dl, x_cv, val_dl, Y_test):
        """
        Train and evaluate a deep learning model.
        :param n_epochs: Number of training epochs
        :param model: The deep learning model
        :param train_dl: Training data loader
        :param x_cv: Validation data
        :param val_dl: Validation data loader
        :param Y_test: True labels for validation data
        """
        batch_size = 50
        no_of_classes = 5
        loss_fn = nn.CrossEntropyLoss()  # Loss Function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer

        train_loss = []
        valid_loss = []

        for epoch in range(n_epochs):
            start_time = time.time()

            # Set model to train configuration
            model.train()  # Indicator for training
            avg_loss = 0.

            for i, (x_batch, y_batch) in enumerate(train_dl):
                # Predict/Forward Pass
                y_pred = model(x_batch)

                # Compute loss
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_dl)

            train_loss, valid_loss, val_accuracy, elapsed_time = self.evaluvation(
                Y_test, avg_loss, batch_size, epoch, loss_fn, model, n_epochs, no_of_classes, start_time, train_loss, val_dl, valid_loss, x_cv)

    def evaluvation(self, Y_test, avg_loss, batch_size, epoch, loss_fn, model, n_epochs, no_of_classes, start_time,
                    train_loss, val_dl, valid_loss, x_cv):
        """
        Evaluate the model's performance on the validation set.
        :param Y_test: True labels for the validation data
        :param avg_loss: Average training loss for the epoch
        :param batch_size: Batch size
        :param epoch: Current epoch
        :param loss_fn: Loss function
        :param model: The deep learning model
        :param n_epochs: Total number of training epochs
        :param no_of_classes: Number of output classes
        :param start_time: Start time for the epoch
        :param train_loss: List to store training losses
        :param val_dl: Validation data loader
        :param valid_loss: List to store validation losses
        :param x_cv: Validation data
        :return: Updated training and validation loss lists, validation accuracy, and elapsed time
        """
        # Set model to validation configuration
        model.eval()  # Indicator for Validation
        avg_val_loss = 0.
        val_preds = np.zeros((len(x_cv), no_of_classes))

        for i, (x_batch, y_batch) in enumerate(val_dl):
            y_pred = model(x_batch).detach()
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(val_dl)

            # Keep/store predictions
            val_preds[i * batch_size:(i + 1) * batch_size] = F.softmax(y_pred).cpu().numpy()

        val_accuracy = sum(val_preds.argmax(axis=1) == Y_test) / len(Y_test)
        train_loss.append(avg_loss)
        valid_loss.append(avg_val_loss)
        elapsed_time = time.time() - start_time

        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(
            epoch + 1, n_epochs, avg_loss, avg_val_loss, val_accuracy, elapsed_time))

        return train_loss, valid_loss, val_accuracy, elapsed_time
