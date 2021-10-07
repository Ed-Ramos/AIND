import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
import PIL
from PIL import Image
import numpy as np


def model_build(arch, epochs, gpu, h1, h2, learning_rate, trainloader, testloader, validloader):
    
    if arch == 'vgg':
        model = models.vgg16(pretrained=True)
        in_features = 25088
        
    else:
        model = models.resnet50(pretrained=True)
        in_features = 2048
                                
    print(model)
                                
    # Use GPU if it's available and chosen
    device = torch.device("cuda" if gpu else "cpu")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(nn.Linear(in_features, h1),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(h1, h2),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(h2, 102),
                           nn.LogSoftmax(dim=1))
                          
    criterion = nn.NLLLoss()
    
    if arch == 'vgg':
        model.classifier = classifier
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        
    #  since only other option is resnet, not need to specify  
    else:
        model.fc = classifier
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
        

    model.to(device);
    
    print(model)
                                
    # Train network and perform validation
    epochs = epochs
    running_loss = 0

    for epoch in range(epochs):
        for inputs, labels in trainloader:
        
         # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            valid_loss = 0
            accuracy = 0
            model.eval()
        
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
                    
            valid_loss += batch_loss.item()
                    
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
        print(f"Epoch {epoch+1}/{epochs}.. "
            f"Train loss: {running_loss/epochs:.3f}.. "
            f"Validation loss: {valid_loss/len(validloader):.3f}.. "
            f"Validation accuracy: {accuracy/len(validloader):.3f}")
        running_loss = 0
        model.train()
        
        
    #Do validation on the test set
    test_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
    
        for inputs, labels in testloader:          
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
                    
            test_loss += batch_loss.item()
                    
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
            print(f"Testing loss: {test_loss/len(testloader):.3f}.. "
                  f"Testing accuracy: {accuracy/len(testloader):.3f}")
        
    return model
    
    
def model_save(train_data, model, model_arch, h1, h2, save_dir):
    
    # Save the checkpoint 

    # map classes to index using train data
    class_to_idx = train_data.class_to_idx

    # invert dictionary and attach to model
    model.class_to_idx = {class_to_idx[k]: k for k in class_to_idx}
    
    if model_arch == "vgg":
        input_size = 25088
        checkpoint_name = '/checkpoint_vgg.pth'
    else:
        input_size = 2048
        checkpoint_name = '/checkpoint_resnet.pth'
        
   
    checkpoint_path = save_dir + checkpoint_name
    
    checkpoint = {'input_size': input_size,
              'output_size': 102,
              'hidden_layers': [h1, h2],
              'arch': model_arch,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict()}
    torch.save(checkpoint, checkpoint_path)
    
    return 