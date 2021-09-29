
def model_build(arch, epochs, gpu, h1, h2, learning_rate):
    
    if arch == vgg:
        model = models.vgg16(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)
                                
    model
    
                                
    # Use GPU if it's available
    device = torch.device("cuda" if gpu else "cpu")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(nn.Linear(25088, h1),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(h1, h2),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(h2, 102),
                           nn.LogSoftmax(dim=1))
                          
    
    model.classifier = classifier

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device);
    
    model
                                
    return