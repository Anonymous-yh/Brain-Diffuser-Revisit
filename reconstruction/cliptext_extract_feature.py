"""
This script extracts CLIP-Text features from image-related captions for a specific subject in 
the NSD (Natural Scenes Dataset). It uses the text encoder from a pretrained Versatile Diffusion (VD) model 
to encode captions into feature vectors.

runnning: python reconstruction/cliptext_extract_feature.py -sub x
"""

import sys
sys.path.append('versatile_diffusion')
import os
import numpy as np

import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
import torchvision.transforms as T

import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Extract CLIP-Text Features')
    parser.add_argument("-sub", "--sub", help="Subject Number", default=1, type=int)
    args = parser.parse_args()
    assert args.sub in [1, 2, 5, 7], "Invalid subject ID"
    return args.sub

def load_model():
    cfgm_name = 'vd_noema'
    pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
    cfgm = model_cfg_bank()(cfgm_name)
    net = get_model()(cfgm)
    sd = torch.load(pth, map_location='cpu')
    net.load_state_dict(sd, strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.clip = net.clip.to(device)
    return net

def extract_clip_features(net, captions, save_path):
    num_samples = len(captions)
    features = np.zeros((num_samples, 77, 768))
    with torch.no_grad():
        for i, annots in enumerate(tqdm(captions, desc=f"Encoding {save_path.split('/')[-1]}")):
            valid_texts = list(annots[annots != ''])
            if len(valid_texts) == 0:
                continue
            c = net.clip_encode_text(valid_texts)
            features[i] = c.to('cpu').numpy().mean(0)
    np.save(save_path, features)

def main():
    sub = parse_args()
    print("Loading Model:")
    net = load_model()

    train_path = f'data/processed_data/subj{sub:02d}/nsd_train_cap_sub{sub}.npy'
    test_path  = f'data/processed_data/subj{sub:02d}/nsd_test_cap_sub{sub}.npy'
    train_caps = np.load(train_path)
    test_caps  = np.load(test_path)

    # os.makedirs(f'data/extracted_features/subj{sub:02d}', exist_ok=True)
    extract_clip_features(net, test_caps,  f'data/extracted_features/subj{sub:02d}/nsd_cliptext_test.npy')
    extract_clip_features(net, train_caps, f'data/extracted_features/subj{sub:02d}/nsd_cliptext_train.npy')

if __name__ == "__main__":
    main()
