

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import PIL
from PIL import Image
import numpy as np
from get_input_args import get_input_args
from utility import data_load
import json
from net_build_train import model_build




# Main program function defined below-
def main():
    
    # get the input arguments
    in_args = get_input_args()

    print(in_args)
    print(type(in_args.gpu))
    
    # load data and define dataloaders
    utility.data_load(in_args.dir)
    
    # generate the cat_to_name dictionary
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    #generate model
    net_build_train.model_build(in_args.arch, in_args.epochs, in_args.gpu, in_args.h1, in_args.h2, in_args.learning_rate)


# Call to main function to run the program
if __name__ == "__main__":
    main()