import json

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms

import helpers
from train import Classifier


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = helpers.get_pretrained_model(checkpoint['pre_trained_model_name'])
    model.classifier = Classifier(checkpoint['classifier_input_size'],
                                  checkpoint['classifier_hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    model.idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
    return model


def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    loaded_model = load_checkpoint(model)
    print(loaded_model)
    loaded_model.to(device)
    im = Image.open(image_path)
    image = helpers.process_image(im)
    image = torch.from_numpy(image).float().to(device)
    image.unsqueeze_(0)
    with torch.no_grad():
        log_ps = loaded_model.forward(image)
        ps = torch.exp(log_ps)
        top_p, top_class_idx = ps.topk(topk, dim=1)

    top_classes = [
        loaded_model.idx_to_class[id]
        for id in top_class_idx.cpu().data.numpy().squeeze()
    ]
    top_p = top_p.cpu().data.numpy().squeeze()
    return top_p, top_classes


def main():
    args = helpers.parse_prediction_arg()
    device = helpers.get_training_device(args.gpu)
    top_p, top_class = predict(args.image_path, args.checkpoint_path,
                               args.top_k, device)
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_class = [cat_to_name[cat] for cat in top_class]
    print(top_p, top_class)


if __name__ == "__main__":
    main()
