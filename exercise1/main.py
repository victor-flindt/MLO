import argparse
import sys
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import torch

from data import mnist
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train, _ = mnist()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        epochs = 1
        steps = 0

        for e in range(epochs):
            tot_train_loss = 0
            for images, labels in train:
                optimizer.zero_grad()
                
                log_ps = model.forward(images.float())
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                tot_train_loss += loss.item()
            print("loss: ", tot_train_loss / 5000)


        
    # def evaluate(self):
    #     print("Evaluating until hitting the ceiling")
    #     parser = argparse.ArgumentParser(description='Training arguments')
    #     parser.add_argument('load_model_from', default="")
    #     # add any additional argument that you want
    #     args = parser.parse_args(sys.argv[2:])
    #     print(args)
        
    #     # TODO: Implement evaluation logic here
    #     model = torch.load(args.load_model_from)
    #     _, test_set = mnist()

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    