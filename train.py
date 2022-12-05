#!/usr/bin/env python3

import argparse

import torch
from torch import optim, nn
import torch.nn.functional as F

from dataloader import load_data
from model import create_model, train_model, save_checkpoint

def init_args():

    parser = argparse.ArgumentParser("The parameters of train script:")

    parser.add_argument(
        "data_dir",
        type=str,
        default="flowers",
        help = "Path to the folder of flower images"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help = "Saving path of the model checkpoints"
    )

    parser.add_argument(
        '--arch',
        type=str,
        default="vgg13",
        choices=["vgg13", "resnet50", "densenet121"],
        help='The model architecture to use'
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for the training process"
    )

    parser.add_argument(
        "--hidden_units",
        type=int,
        default=512,
        help="Number of the hidden units for the model architecture"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of the epochs for the training process"
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Flag for training with gpu instead of cpu"
    )

    return parser.parse_args()

def main():
    in_arg = init_args()

    trainloader, validloader, testloader = load_data(in_arg.data_dir)

    model = create_model(in_arg.arch,in_arg.hidden_units)

    device = torch.device("cuda" if (torch.cuda.is_available()  and in_arg.gpu) else "cpu")

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(
        model.fc.parameters() if in_arg.arch=="resnet50" else model.classifier.parameters(),
        lr=in_arg.learning_rate
    )

    model.to(device)

    print(f"Training ({device})...")
    epoch, steps=train_model(in_arg.epochs, trainloader, validloader, device, model, optimizer, criterion)

    model.cpu()
    save_checkpoint(in_arg.save_dir+'checkpoint.pth', model, optimizer, epoch, steps, in_arg.arch, in_arg.hidden_units, in_arg.learning_rate)
    
    print("Finished")

if __name__ == "__main__":
    main()
