

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import PIL
from PIL import Image
import numpy as np
from get_input_args import get_train_input_args
from utility import data_load
import json
from net_build_train import model_build, model_save


# Main program function defined below-
def main():
    
    # get the input arguments
    in_args = get_train_input_args()
    
    # load data and define dataloaders
    trainloader, testloader, validloader, train_data = data_load()

    #generate model
    trained_model = model_build(in_args.arch, in_args.epochs, in_args.gpu, in_args.h1, in_args.h2, in_args.learning_rate,trainloader, testloader, validloader)
    
    model_save(train_data, trained_model, in_args.arch, in_args.h1, in_args.h2, in_args.save_dir)
    

# Call to main function to run the program
if __name__ == "__main__":
    main()