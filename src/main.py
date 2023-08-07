import torch
from torchvision import datasets, transforms
from torch import optim, nn
from Network import myNetwork, run_test, initiate_training


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    ])
    # set hyperparameters
    batch_sz = 64
    learning_rate = 0.01
    momentum = 0.5

    # downloading and loading the MNIST database
    # already downloaded before so set download to false if running again
    training_set = datasets.MNIST("./data", download=False, train=True, transform=transform)
    test_set = datasets.MNIST("./data", download=False, train=False, transform=transform)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_sz, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_sz, shuffle=True)

    # initialize Neural Network
    model = myNetwork()

    # initialize loss function
    # this "negative log-likelihood loss" function
    # works in conjunction with the LogSoftmax function
    # we use to compute values for the output layer
    criterion = nn.NLLLoss()

    # initialize optimizer
    # handles backprop and updating weights and biases
    # using stochastic gradient descent
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # start training the model
    # refer to Network.py
    initiate_training(model, 15, training_loader, criterion, optimizer)

    # run test on model
    # refer to Network.py
    run_test(model, test_loader)

