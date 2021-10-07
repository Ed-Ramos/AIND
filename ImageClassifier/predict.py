
import torch
import json
from get_input_args import get_predict_input_args
from utility import data_load, load_checkpoint, process_image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #model set to evaluation mode so that layers like Dropout layer can work correctly. 
    #Prevents inputs from being dropped randomlywhen making predictions.
    model.eval()
    
    processed_image = process_image(image_path)
    processed_image.unsqueeze_(0)   
    log_ps = model(processed_image)
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(topk, dim=1)
    
    #convert to a list
    probs_list = top_p.detach().numpy().squeeze().tolist()
    classes_list = top_class.detach().numpy().squeeze().tolist()
    
    #for a single element, need to convert to list as above code outputs a python scalar
    if not isinstance(classes_list, list):
        classes_list = [classes_list]
        
    if not isinstance(probs_list, list):
        probs_list = [probs_list]
    
    return probs_list, classes_list

def get_class_names(classes, model, category_path):
    
    # generate the cat_to_name dictionary
    with open(category_path, 'r') as f:
        cat_to_name = json.load(f)
    
    # create a list of flower class names that correspond to the probable numerical classes determined by predict function
    class_names = []
    for i in range(len(classes)):
        class_names.append(cat_to_name[model.class_to_idx[classes[i]]])
    return class_names

# Main program function defined below-
def main():
    
    # get the input arguments
    in_args = get_predict_input_args()
    print(in_args)
    
    model = load_checkpoint(in_args.checkpoint, in_args.gpu)
    probs, classes = predict(in_args.image, model, in_args.top_k)
    class_names = get_class_names(classes, model, in_args.category_names)
    
    print('Top probabilities are ' + str(probs))
    print('Top classes are ' + str(class_names))

# Call to main function to run the program
if __name__ == "__main__":
    main()
