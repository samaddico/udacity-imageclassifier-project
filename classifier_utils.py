import torch

from os import path
from torchvision import models

__FILENAME="model_checkpoint.py"

def save_checkpoint(model, save_dir, train_data):
    """
    """
    if type(save_dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if path.isdir(save_dir):
            # Create `class_to_idx` attribute in model
            model.class_to_idx = train_data.class_to_idx
            
            # Create checkpoint dictionary
            checkpoint = {
                'architecture': model.name,
                'classifier': model.classifier,
                'class_to_idx': model.class_to_idx,
                'state_dict': model.state_dict()
            }
            
            # Save checkpoint
            torch.save(checkpoint, path.join(save_dir, __FILENAME))
        else: 
            print("Directory not found, model will not be saved.")

def load_checkpoint(checkpoint):
    """
    Loads a saved trained model from file
    
    :param checkpoint: file to read from
    """
    checkpoint = torch.load(checkpoint)
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['architecture'])
        model.name = checkpoint['architecture']
    
    # freeze parameters
    for param in model.parameters(): param.requires_grad = False
    
    # Load checkpoint details
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def check_gpu(gpu):
    """
    Checks which device type to use depending on user gpu input
    If `gpu` is False then it returns the cpu device
    :param gpu: gpu  type provide on commandline by user
    :return: device 
    """
   # 
    if not gpu:
        return torch.device("cpu")
    
    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Print result
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device