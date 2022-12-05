#!/usr/bin/env python3

import argparse

import torch
from torch import optim
import torch.nn.functional as F

import numpy as np

from dataloader import load_categories
from model import load_checkpoint, predict_model

def init_args():

    parser = argparse.ArgumentParser("The parameters of test script:")

    parser.add_argument(
        "image_path",
        type=str,
        default="assets/Flowers.png",
        help = "Path to a single flower image for the testing process"
    )

    parser.add_argument(
        "checkpoint",
        type=str,
        default="checkpoints",
        help = "Loading path of the model checkpoint"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of the top probalities for the testing process"
    )

    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
        help = "Path for the class category file"
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Flag for testing with gpu instead of cpu"
    )

    return parser.parse_args()

def main():
    in_arg = init_args()

    model, optimizer, epoch, steps = load_checkpoint(in_arg.checkpoint+'checkpoint.pth')

    device = torch.device("cuda" if (torch.cuda.is_available()  and in_arg.gpu) else "cpu")

    model.to(device)
    model.eval()

    prob, cls = predict_model(in_arg.image_path, device, model, in_arg.top_k)

    probs = prob.cpu().detach().numpy()[0].tolist()
    classes = cls.cpu().detach().numpy()[0].tolist()
   
    df_cls=load_categories(in_arg.category_names)

    class_texts=[]
    for c in classes:
        class_texts.append(df_cls.loc[str(c)]["class"])

    print(f"The top {in_arg.top_k} result(s):")
    for p, c in zip(probs, class_texts):
        print(f"Probality: {p:.3f}, Class: {c}")

if __name__ == "__main__":
    main()
