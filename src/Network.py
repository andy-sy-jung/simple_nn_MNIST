import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class myNetwork(nn.Module):
    def __init__(self):
        # 4 total layers: input layer of 784 nodes,
        # 2 hidden layers of 164 and 64 nodes
        # and output layer of 10 nodes
        super(myNetwork, self).__init__()

        # input layer maps from 784 to 128 nodes
        self.linear1 = nn.Linear(784, 256)
        # first activation using Rectified Linear Unit (ReLU)
        self.activation1 = nn.ReLU()
        # hidden layer 1 maps from 128 nodes to 64 nodes
        self.linear2 = nn.Linear(256, 128)
        # second activation by ReLU again
        self.activation2 = nn.ReLU()
        # hidden layer 2 maps from 64 nodes to output layer of 10 nodes
        self.linear3 = nn.Linear(128, 10)
        # convert values to log probabilities using LogSoftmax
        self.softmax = nn.LogSoftmax(dim=1)

    # defining the forward propagation method,
    # basically calling all the linear regression
    # and activation functions
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x


# Core training of the neural network:
# For each batch of 64 pictures and labels in the iterable training_loader
# run through forward prop, then calculate the gradient, then update weights, then repeat
def initiate_training(model, epochs, training_loader, criterion, optimizer):

    for e in range(epochs):
        running_loss = 0
        for images, labels in training_loader:
            # have to "flatten" the images into 64, 784 long vector
            images = images.view(images.shape[0], -1)

            # empty the gradients for each loop
            optimizer.zero_grad()

            # run the images through the model
            output = model(images)
            # calculate loss using the loss function
            # comparing the predicted and the correct labels
            loss = criterion(output, labels)

            # back propagate using .backward()
            # to calculate the gradient
            loss.backward()

            # update the weights and biases
            optimizer.step()

            # show the calculated loss each epoch
            running_loss += loss.item()
        else:
            print("Epoch {} - Training Loss: {}".format(e, running_loss/len(training_loader)))


def run_test(model, test_loader):
    correct_count, all_count = 0, 0
    for images, labels in test_loader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)

            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if pred_label == true_label:
                correct_count += 1
            all_count += 1

    print("\nNumber of Images Tested:", all_count)
    print("Model Accuracy:", (correct_count/all_count))


#def test_random_number(model, test_loader):






