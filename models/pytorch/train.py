import os
import sys
import numpy as np
import pandas as pd

import torch 
import torchvision


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y).sum()

def evaluate_acc(test_dataload):
    
    accs = []
    lengs = []

    for batch_X, batch_y in test_dataload:
        
        batch_y_hat = model(batch_X)
        acc = (batch_y_hat.argmax(axis=1) == batch_y).sum()
        leng = len(batch_y_hat)
        
        accs.append(acc)
        lengs.append(leng)
        
    return sum(accs)/sum(lengs)

def train_epoch(train_dataload):

    for batch_X, batch_y in train_dataload:
        
        batch_y_hat = model(batch_X)
        batch_loss = loss(batch_y_hat, batch_y)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

def train(epoch_num):
    for i in range(epoch_num):
        train_epoch(train_dataload)
        accuracy = evaluate_acc(test_dataload)
        print("Epoch {}, Accuracy: {:.2f}%".format(i, accuracy*100))
        
        


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--model_dir', type=str, default='./pytorch_model.pkl')
    args = parser.parse_args()
    print(args.batch_size)


    train_data = torchvision.datasets.MNIST(root="./../0_sample_data/", train=True, transform=torchvision.transforms.ToTensor(), download=False)
    test_data = torchvision.datasets.MNIST(root="./../0_sample_data/", train=False, transform=torchvision.transforms.ToTensor(), download=False)
    train_dataload = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    test_dataload = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=args.batch_size)

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    )
    model.apply(init_weights)

    loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adagrad(params=model.parameters(), lr=args.learning_rate)

    train(epoch_num=args.epochs)
    
    torch.save(model.state_dict(), args.model_dir)
    
    sys.exit(0)
