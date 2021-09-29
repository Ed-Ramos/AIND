

import argparse


def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 6 command line arguments. If 
    the user fails to provide some or all of the 6 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as --dir with default value 'pet_images'
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. Hiden layer 1 as --h1 with default of 4096
      4. Hiden layer 2 as --h2 with default of 512
      5. Learning Rate as --learning_rate with default value of 0.001
      6. Epochs as --epochs with a default value of 10
      7. GPU as --gpu with a default of false
      
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    
    # Argument 1: that's a path to a folder
    parser.add_argument('--dir', type = str, default = 'flowers', 
                    help = 'path to the folder of images') 
    
    # Argument 2: Choose a model. Two choices allowed
    parser.add_argument('--arch', type = str, choices=['vgg', 'resnet'], default = 'vgg', 
                    help = 'cnn model architecture') 
    
    # Argument 3: Choose hidden layer 1 units. 
    parser.add_argument('--h1', type = int, default = 4096, 
                    help = 'number of hidden units for layer one') 
    
    # Argument 4: Choose hidden layer 2 units. 
    parser.add_argument('--h2', type = int, default = 512, 
                    help = 'number of hidden units for layer two') 
    
     # Argument 5: Choose learning rate from a choice of 3
    parser.add_argument('--learning_rate', type = float, choices=[0.1, 0.01, 0.001], default = 0.001, 
                    help = 'leaning rate') 

    # Argument 6: Choose number of training epochs. Max of 10. 
    parser.add_argument('--epochs', type = int, choices=range(1, 11), default = 8, 
                    help = 'number of training epochs. Max of 10') 
    
    # Argument 7: Indicate if GPU is available
    parser.add_argument('--gpu', action = 'store_true', 
                    help = 'indicate if GPU is available') 
    
    
    
    return parser.parse_args()
