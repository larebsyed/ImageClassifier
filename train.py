#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms

import helpers


class Classifier(nn.Module):
    def __init__(self, input_features, hidden_units):
        super().__init__()
        self.fc0 = nn.Linear(input_features, hidden_units, bias=True)
        self.fc1 = nn.Linear(hidden_units, 512, bias=True)
        self.fc4 = nn.Linear(512, 256, bias=True)
        self.fc5 = nn.Linear(256, 102, bias=True)
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # Now with dropout
        x = self.dropout(F.relu(self.fc0(x)))

        # output so no dropout here
        x = F.log_softmax(self.fc1(x), dim=1)
        return x


def get_model(hidden_units, arch):
    pretrained_model = helpers.get_pretrained_model(arch)
    classifier_input_features = helpers.get_pretrained_classifier_input_features(
        arch)
    pretrained_model.classifier = Classifier(classifier_input_features,
                                             hidden_units)
    return pretrained_model


def train_model(model, epochs, trainloader, validloader, optimizer, criterion,
                device):
    model.to(device)
    for e in range(epochs):
        running_loss = 0
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    log_ps = model.forward(inputs)
                    test_loss += criterion(log_ps, labels)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            print(
                "Epoch: {}/{}.. ".format(e + 1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss /
                                                  len(trainloader)),
                "Test Loss: {:.3f}.. ".format(test_loss / len(validloader)),
                "Test Accuracy: {:.3f}".format(
                    (accuracy / len(validloader) * 100)))


def load_dataset(data_dir):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_validation_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_datasets = datasets.ImageFolder(valid_dir,
                                          transform=test_validation_transform)
    test_datasets = datasets.ImageFolder(test_dir,
                                         transform=test_validation_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets,
                                              batch_size=64,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets,
                                             batch_size=64,
                                             shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets,
                                              batch_size=64,
                                              shuffle=True)
    return trainloader, validloader, testloader, train_datasets.class_to_idx


def main():
    args = helpers.parse_training_args()
    print("Running with args", args)
    model = get_model(args.hidden_units, args.arch)
    print(model)
    criterion = nn.NLLLoss()
    device = helpers.get_training_device(args.gpu)
    optimizer = optim.Adam(model.classifier.parameters(),
                           lr=args.learning_rate)
    trainloader, validloader, testloader, class_to_idx = load_dataset(
        args.data_dir)
    train_model(model, args.epochs, trainloader, validloader, optimizer,
                criterion, device)
    helpers.store_model_checkpoint(model, class_to_idx, args.epochs, args.arch,
                                   args.hidden_units, args.epochs, optimizer,
                                   args.save_dir)


if __name__ == "__main__":
    main()
