import argparse

def get_train_input_args():
    """
    Gets commandline inputs for training module
    """
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help = "Provide data directory. Mandatory argument", type = str)
    parser.add_argument("--save_dir", type=str, default=".",
                        help="directory where to save trained model and details")
    parser.add_argument("--arch", type=str, default="vgg",
                        help="pre-trained model: vgg16, resnet18, alexnet")
    parser.add_argument("--gpu", type=str, default="cpu",
                        help="Use GPU + Cuda for calculations")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                        help="number of epochs to train model")
    parser.add_argument("--hidden_units", type=int, default=4096,
                        help="hidden layer")
    return parser.parse_args()

def get_predict_input_args():
    """
    Gets commandline inputs for predict module
    """
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="path to image in which to predict class label")
    parser.add_argument("checkpoint", type=str, help="checkpoint in which trained model is contained")
    parser.add_argument("--top_k", type=int, default=5, help="number of classes to predict")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json",
                        help="file to convert label index to label names")
    parser.add_argument("--gpu",type=str, default="cpu", help="Use GPU + Cuda for calculations")
    
    return parser.parse_args()