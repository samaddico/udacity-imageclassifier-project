import torch
import numpy as np
import pandas as pd
import seaborn as sb

from PIL import Image
from torch import nn, optim
from collections import OrderedDict
from torchvision import datasets, transforms, models

from get_input_args import get_train_input_args
from classifier_utils import save_checkpoint
from classifier_utils import check_gpu

# load pretrained models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

def main():
    
    # get commandline inputs
    in_arg = get_train_input_args()
    
    # load and process images from data directory
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # data loaders
    train_data_loader, valid_data_loader, test_data_loader, train_image_datasets = build_transforms(
        train_dir, 
        valid_dir, 
        test_dir
    )
    
    # load the pretrained model and determine input size
    model_dict = {"vgg": vgg16, "resnet": resnet18, "alexnet": alexnet}
    inputsize_dict = {"vgg": 25088, "resnet": 512, "alexnet": 9216}
    model_name = {"vgg": "vgg16", "resnet": "resnet18", "alexnet": "alexnet"}
    
    model = model_dict[in_arg.arch]
    input_size = inputsize_dict[in_arg.arch]
    
    # Freeze Parameters
    for param in model.parameters():
        param.requires_grad = False
    
    input_features = model.classifier[0].in_features
    
    # update classifier parameters
    classifier = nn.Sequential(OrderedDict([
       ('fc1', nn.Linear(input_features, in_arg.hidden_units, bias=True)),
       ('relu1', nn.ReLU()),
       ('dropout1', nn.Dropout(p=0.5)),
       ('fc2', nn.Linear(in_arg.hidden_units, 102, bias=True)),
       ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)    
    
    # get device
    device = check_gpu(in_arg.gpu)
    
    # Send model to device
    model.to(device);
    
    trained_model = train_model(
        model, 
        device,
        train_data_loader, 
        valid_data_loader, 
        test_data_loader,
        criterion, 
        optimizer, 
        in_arg.epochs, 
        in_arg.gpu
    )
    
    # set model name
    model.name = model_name[in_arg.arch]
    
    # compute accuracy of model
    model_accuracy = get_model_accuracy(test_data_loader, trained_model, criterion)
    print(f"Accuracy is : {model_accuracy}")
    
    # save pre trained model to file 
    save_checkpoint(trained_model, in_arg.save_dir, train_image_datasets)
    
    
def build_transforms(train_dir, valid_dir, test_dir):
    train_data_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(244),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    valid_data_transforms = transforms.Compose ([
        transforms.Resize (255),
        transforms.CenterCrop (224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    test_data_transforms = transforms.Compose ([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    
    # Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform = train_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform = valid_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform = test_data_transforms)

    # define the dataloaders
    train_data_loader = torch.utils.data.DataLoader(train_image_datasets, batch_size = 64, shuffle = True)
    valid_data_loader = torch.utils.data.DataLoader(valid_image_datasets, batch_size = 64, shuffle = True)
    test_data_loader = torch.utils.data.DataLoader(test_image_datasets, batch_size = 64, shuffle = True)

    return train_data_loader, valid_data_loader, test_data_loader, train_image_datasets

def validate_model(model, valid_loader, criterion):
    """
    Validate model performance
    """
    model.to('cuda')
    
    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:
        
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

def train_model(model, device, train_data_loader, valid_data_loader, test_data_loader,
                criterion, optimizer, epochs, gpu):
    print("training process starting .....\n")

    print_every = 30 # Prints every 30 images out of batch of 50 images
    steps = 0
    for e in range(epochs):
        running_loss = 0
        model.train() 

        for ii, (inputs, labels) in enumerate(train_data_loader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validate_model(model, valid_data_loader, criterion)

                print("Epoch: {}/{} | ".format(e+1, epochs),
                      "Training Loss: {:.4f} | ".format(running_loss/print_every),
                      "Validation Loss: {:.4f} | ".format(valid_loss/len(test_data_loader)),
                      "Validation Accuracy: {:.4f}".format(accuracy/len(test_data_loader)))

                running_loss = 0
                model.train()

    print("\ntraining process is now complete!!")
    return model

def get_model_accuracy(test_loader, model, criterion):
    """
    Computes how accurate the trained model predicts data it was not trained on
    """
    correct_matches = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')

            # forward
            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            # tensor with probability, tensor with index of flower category
            probability = torch.exp(output) # tensor with prob. of each flower category
            prediction = probability.max(dim=1) # tensor giving us flower label most likely

            # calculate the number of correct matches
            matches = (prediction[1] == labels.data)
            correct_matches += matches.sum().item()
            total += 64

        accuracy_percentage = 100 * (correct_matches / total)
        return accuracy_percentage    

# run application    
if __name__ == '__main__': main()   