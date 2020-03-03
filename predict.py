import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from math import ceil

from torchvision import models, transforms

from get_input_args import get_predict_input_args
from classifier_utils import load_checkpoint
from classifier_utils import check_gpu

def main():
    # get inputs
    in_arg = get_predict_input_args()
    
    # load json data
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # load trained model
    model = load_checkpoint(in_arg.checkpoint)
    
    # process image
    processed_image = process_image(in_arg.image_path)
    
    # get device
    device = check_gpu(in_arg.gpu)
    model.to(device)
    
    # do prediction
    probs, labs, flowers = predict(
        model, 
        in_arg.image_path, 
        cat_to_name, 
        in_arg.top_k
    )
    
    # display results
    print_prediction_result(probs, flowers)
    # display results
#     display_result_graph(
#         model, 
#         in_arg.image_path, 
#         cat_to_name, 
#         flowers
#     )
    
def process_image(image):
    """ 
    Scales, crops, and normalizes a PIL image for a PyTorch model
    :param image: image to be processed
    :return an Numpy array
    """
    
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_transformed = transform(image)
    return np.array(image_transformed)

def imshow(image, ax=None, title=None):
    """
    Imshow for Tensor.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(model, image_path, cat_to_name, topk=5):
    """ 
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    # No need for GPU on this part (just causes problems)
    model.to("cpu")
    
    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    # Find probabilities (results) by passing through the function 
    # (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(topk)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers

def display_result_graph(model, image_path, cat_to_name, flowers):
    """
    Displays a graph of the top `tok_k' prediction results
    
    :param model: trained model
    :param image_path: image input
    :param flowers: list of flowers
    """
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)

    # Set up title
    flower_number = image_path.split('/')[2]
    title_ = cat_to_name[flower_number]

    # Plot flower
    image = process_image(image_path)
    imshow(image, ax, title = title_);

    # Plot bar chart
    plt.subplot(2,1,2)
    sb.barplot(x=probs, y=flowers, color=sb.color_palette()[0]);
    plt.show()
    
def print_prediction_result(probs, flowers):
    """
    Converts two lists into a dictionary to print on screen
    
    :param probs: the list of probabilities
    :param flowers: list of flowers
    """
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, probability: {}%".format(j[0], ceil(j[1]*100)))

        
# run application  
if __name__ == '__main__': main()