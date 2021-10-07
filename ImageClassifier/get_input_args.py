
import argparse

def get_train_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 7 command line arguments. If 
    the user fails to provide some or all of the 7 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Checkpoint Folder as --save_dir with default value './checkpoints'
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. Hiden layer 1 as --h1 with default of 1024
      4. Hiden layer 2 as --h2 with default of 512
      5. Learning Rate as --learning_rate with default value of 0.001
      6. Epochs as --epochs with a default value of 10
      7. GPU as --gpu with a default of True
      
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Argument 1: Path to directory to save checkpoint
    parser.add_argument('--save_dir', type = str, default = './checkpoints', 
                    help = 'path to directory to save checkpoint') 
    
    # Argument 2: Choose a model. Two choices allowed
    parser.add_argument('--arch', type = str, choices=['vgg', 'resnet'], default = 'vgg', 
                    help = 'cnn model architecture') 
    
    # Argument 3: Choose hidden layer 1 units. 
    parser.add_argument('--h1', type = int, default = 1024, 
                    help = 'number of hidden units for layer one') 
    
    # Argument 4: Choose hidden layer 2 units. 
    parser.add_argument('--h2', type = int, default = 512, 
                    help = 'number of hidden units for layer two') 
    
    # Argument 5: Choose learning rate from a choice of 3
    parser.add_argument('--learning_rate', type = float, choices=[0.1, 0.01, 0.001], default = 0.001, 
                    help = 'leaning rate') 

    # Argument 6: Choose number of training epochs. Max of 10. 
    parser.add_argument('--epochs', type = int, choices=range(1, 11), default = 6, 
                    help = 'number of training epochs. Max of 10') 
    
    # Argument 7: Indicate if GPU is is chosen for training 
    parser.add_argument('--gpu', type = str, choices=[True, False], default = True, 
                    help = 'indicate if GPU is chose for training') 
    
    return parser.parse_args()



def get_predict_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 5 command line arguments. If 
    the user fails to provide some or all of the 5 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      
      1. GPU as --gpu with a default of True
      2. Path to checkpoint file as --checkpoint with default of ./checkpoints/checkpoint_vgg.pth
      3. Top k most lilely classes as --top_k with default of 1
      4. Path to an image to predict as --image with default ./flowers/test/78/image_01888.jpg
      5. Path to category to name file as --category_names with default cat_to_name.json
      
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    
    
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()
    
    # Argument 1: Indicate if want to use GPU is calculate predictions
    parser.add_argument('--gpu', type = str, choices=['True', 'False'], default = 'True', 
                    help = 'Indicate if want to use GPU is calculate predictions') 
    
    # Argument 2: Path to a checkpoint file
    parser.add_argument('--checkpoint', type = str, default = './checkpoints/checkpoint_vgg.pth', 
                    help = 'path to a checkpoint file')
    
    # Argument 3: Top k most likely classes along with probablities)
    parser.add_argument('--top_k', type = int, choices=range(1, 11), default = 1, 
                    help = 'Top k most likely classes along with probablities') 
    
    # Argument 4: Path to an image to predict
    parser.add_argument('--image', type = str,  default = './flowers/test/78/image_01888.jpg',
                    help = 'path to an image to predict')
    
    # Argument 5: Path to a category to names json file
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                    help = 'path to a category to names json file')
    
    
    return parser.parse_args()
    
    
