
import torch
from torchvision import datasets, transforms, models
import PIL
from PIL import Image
import numpy as np



def data_load(images_dir):
    
    data_dir = images_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    return


def process_image(image):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    
    #resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    basewidth = None 
    baseheight = None 
    
    if image.size[0] < image.size[1]:
        basewidth = 256
    else:
        baseheight = 256 
    
    if basewidth: 
        wpercent = (basewidth/float(image.size[0]))
        hsize = int((float(image.size[1])*float(wpercent)))
        image = image.resize((basewidth,hsize), Image.ANTIALIAS)
    
    if baseheight: 
        hpercent = (baseheight/float(image.size[1]))
        wsize = int((float(image.size[0])*float(hpercent)))
        image = image.resize((wsize,baseheight), Image.ANTIALIAS)
    
    # getting the width and heith of image
    w,h = image.size
    
    #cropping the center 224x224 portion of image
    image = image.crop(((w - 224)/2, (h - 224)/2, w - (w - 224)/2, h - (h - 224)/2))
    
    #creating a numpy array from image
    image_array = np.array(image)
    
    #encoding channels to floats between (0,1)
    image_array = image_array/256
    
    
    image_means = np.array([0.485, 0.456, 0.406])
    image_std = np.array([0.229, 0.224, 0.225])
    
    #normalizing in specific way
    image_array = (image_array - image_means)/image_std
    
    # reordering dimentions so that color channel is first dimension
    image_array = image_array.transpose((2,0,1))
    
    #converting to tensor
    image_tensor = torch.from_numpy(image_array).type(torch.FloatTensor)
    
    return image_tensor