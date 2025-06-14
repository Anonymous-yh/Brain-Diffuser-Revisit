"""
This script extracts visual features from reconstructed brain images using multiple pretrained neural networks 
(e.g., CLIP, InceptionV3, AlexNet). It supports various reconstruction methods and subjects, and saves the extracted 
features for further evaluation or analysis.

running: python evaluate/evaluate_features.py -sub x -method 'brain-diffuser'

"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as tvmodels
import clip

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument("-sub", "--sub", help="Subject Number", default=1, type=int)
    parser.add_argument("-method", "--method", help="Generate Method", default='brain-diffuser')
    return parser.parse_args()

def get_directories(sub, method):
    images_dir = 'data/nsddata_stimuli/test_images'
    feats_dir = 'data/eval_features/test_images'

    if sub in [1, 2, 5, 7]:
        feats_dir = f'data/eval_features/subj{sub:02d}'
        if method == 'brain-diffuser':
            images_dir = f'results/versatile_diffusion/subj{sub:02d}'
        elif method == 'only-VDVAE':
            images_dir = f'results/vdvae/subj{sub:02d}'
        elif method == 'wo-cliptext':
            images_dir = f'results/versatile_diffusion/without_cliptext/subj{sub:02d}'
        elif method == 'wo-clipvision':
            images_dir = f'results/versatile_diffusion/without_clipvision/subj{sub:02d}'

    os.makedirs(feats_dir, exist_ok=True)
    return images_dir, feats_dir

class ExternalImageDataset(Dataset):
    def __init__(self, data_path, net_name='clip', prefix=''):
        self.data_path = data_path
        self.prefix = prefix
        self.net_name = net_name
        self.num_test = 982

        if self.net_name == 'clip':
            self.normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
        else:
            self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, f"{self.prefix}{idx}.png")
        img = Image.open(img_path)
        img = T.functional.resize(img, (224, 224))
        img = T.functional.to_tensor(img).float()
        img = self.normalize(img)
        return img

    def __len__(self):
        return self.num_test

def register_hook(layer_name, net, fn):
    if layer_name == 'avgpool':
        net.avgpool.register_forward_hook(fn)
    elif layer_name == 'lastconv':
        net.Mixed_7c.register_forward_hook(fn)
    return net

def get_model(net_name, layer, device, fn):
    if net_name == 'inceptionv3':
        net = tvmodels.inception_v3(pretrained=True)
        net = register_hook(layer, net, fn)

    elif net_name == 'alexnet':
        net = tvmodels.alexnet(pretrained=True)
        if layer == 2:
            net.features[4].register_forward_hook(fn)
        elif layer == 5:
            net.features[11].register_forward_hook(fn)
        elif layer == 7:
            net.classifier[5].register_forward_hook(fn)

    elif net_name == 'clip':
        model, _ = clip.load("ViT-L/14", device=f'cuda:{device}')
        net = model.visual.float()
        if layer in [7, 12]:
            net.transformer.resblocks[layer].register_forward_hook(fn)
        elif layer == 'final':
            net.register_forward_hook(fn)

    elif net_name == 'efficientnet':
        net = tvmodels.efficientnet_b1(weights=True)
        net.avgpool.register_forward_hook(fn)

    elif net_name == 'swav':
        net = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        net.avgpool.register_forward_hook(fn)

    return net.eval().to(f'cuda:{device}')

def extract_features(net_name, layer, dataset, device):
    feat_list = []
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    net = get_model(net_name, layer, device, lambda m, i, o: feat_list.append(o.detach().cpu().numpy()))

    with torch.no_grad():
        for x in tqdm(loader, desc=f"Extracting {net_name}-{layer}"):
            x = x.to(f'cuda:{device}')
            _ = net(x)

    if net_name == 'clip' and layer in [7, 12]:
        feat_list = np.concatenate(feat_list, axis=1).transpose((1, 0, 2))
    else:
        feat_list = np.concatenate(feat_list)
    return feat_list

def main():
    args = parse_arguments()
    sub = int(args.sub)
    assert sub in [0, 1, 2, 5, 7], "Subject must be in [0, 1, 2, 5, 7]"

    device = 1
    net_list = [
        ('inceptionv3', 'avgpool'),
        ('clip', 'final'),
        ('alexnet', 2),
        ('alexnet', 5),
        ('efficientnet', 'avgpool'),
        ('swav', 'avgpool')
    ]

    images_dir, feats_dir = get_directories(sub, args.method)

    for net_name, layer in net_list:
        print(f"\nProcessing Network: {net_name}, Layer: {layer}")
        dataset = ExternalImageDataset(data_path=images_dir, net_name=net_name)
        features = extract_features(net_name, layer, dataset, device)
        file_name = os.path.join(feats_dir, f"{net_name}_{layer}_{args.method}.npy")
        np.save(file_name, features)

if __name__ == "__main__":
    main()
