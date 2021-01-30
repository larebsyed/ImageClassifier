import argparse
import os

import numpy as np
import torch
from torchvision import datasets, models, transforms


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("path is not a valid path")


def get_training_device(gpu):
    if not gpu:
        return torch.device('cpu')
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pretrained_model(model_name):
    return {
        'vgg': models.vgg19(pretrained=True),
        'densenet': models.densenet121(pretrained=True),
    }.get(model_name, models.vgg19(pretrained=True))


def get_pretrained_classifier_input_features(model_name):
    return {'vgg': 25088, 'densenet': 1024}.get(model_name, 25088)


def parse_prediction_arg():
    parser = argparse.ArgumentParser(description='Predict with models')
    parser.add_argument('image_path', help='path to image')
    parser.add_argument('checkpoint_path', help='path to checkpoint')
    parser.add_argument('--category_names', help='path to category name file')
    parser.add_argument('--top_k',
                        help='max number classes',
                        type=int,
                        default=3)
    parser.add_argument('--gpu',
                        action='store_true',
                        help='run on gpu',
                        default=False)
    args = parser.parse_args()
    return args


def parse_training_args():
    parser = argparse.ArgumentParser(description='Train the models')
    parser.add_argument('data_dir',
                        type=dir_path,
                        help='path of data directory')
    parser.add_argument('--save-dir',
                        help='path to store the trained model checkpoint',
                        type=dir_path,
                        default='/home/workspace/saved_models')
    parser.add_argument('--arch',
                        choices=('vgg', 'densenet'),
                        help='which pretrained model to use',
                        default='vgg')
    parser.add_argument('--learning-rate',
                        type=float,
                        help='learning rate for model training',
                        default=0.01)
    parser.add_argument('--hidden-units',
                        type=int,
                        help='hidden layers in model classifiers',
                        default=520)
    parser.add_argument('--epochs',
                        type=int,
                        help='total number of training cycle_spin',
                        default=20)
    parser.add_argument('--gpu',
                        action='store_true',
                        help='run on gpu',
                        default=False)
    args = parser.parse_args()
    return args


def store_model_checkpoint(model, class_to_idx, current_epochs, arch,
                           hidden_units, max_epochs, optimizer,
                           checkpoint_path):
    model.class_to_idx = class_to_idx
    checkpoint = {
        'start_epoch': current_epochs,
        'end_epoch': max_epochs,
        'classifier_input_size':
        get_pretrained_classifier_input_features(arch),
        'classifier_output_size': 102,
        'classifier_hidden_units': hidden_units,
        'classifier_dropout_ratio': 0.3,
        'pre_trained_model_name': arch,
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, checkpoint_path + "/model.pth")


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    size = 256, 256
    image.thumbnail(size)
    box = (16, 16, 240, 240)
    im = image.crop(box)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    im = (im - mean) / std
    im = im.transpose((2, 1, 0))

    return im


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
