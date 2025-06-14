"""
This script extracts CLIP-Vision features from stimulus images in the Natural Scenes Dataset (NSD) 
for a specified subject. It processes both training and test images and saves the extracted features 
for downstream analysis (e.g., brain encoding/decoding models).

running: python reconstruction/clipvision_extract_feature.py -sub x
"""
import sys
import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

sys.path.append('versatile_diffusion')
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model

# ----------------------- Dataset Class ---------------------------
class NSDImageDataset(Dataset):
    def __init__(self, npy_path):
        self.images = np.load(npy_path).astype(np.uint8)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        img = T.functional.resize(img, (512, 512))
        img = T.functional.to_tensor(img).float()
        img = img * 2 - 1  # Normalize to [-1, 1]
        return img

    def __len__(self):
        return len(self.images)

# ----------------------- Feature Extraction Function -----------------------
def extract_clip_vision_features(sub: int, net, device, batch_size: int = 1):
    print(f"\n Extracting CLIP-Vision features for subject {sub}...")

    # Data paths
    train_path = f'data/processed_data/subj{sub:02d}/nsd_train_stim_sub{sub}.npy'
    test_path = f'data/processed_data/subj{sub:02d}/nsd_test_stim_sub{sub}.npy'

    # Datasets and loaders
    train_dataset = NSDImageDataset(train_path)
    test_dataset = NSDImageDataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_embed, num_features = 257, 768
    num_train, num_test = len(train_dataset), len(test_dataset)

    train_clip = np.zeros((num_train, num_embed, num_features))
    test_clip = np.zeros((num_test, num_embed, num_features))

    # Create output directory
    out_dir = f'data/extracted_features/subj{sub:02d}'
    os.makedirs(out_dir, exist_ok=True)

    # Feature extraction loop
    with torch.no_grad():
        for i, img_batch in enumerate(tqdm(test_loader, desc="Processing Test Set")):
            img_batch = img_batch.to(device)
            clip_features = net.clip_encode_vision(img_batch)
            test_clip[i] = clip_features[0].cpu().numpy()

        np.save(os.path.join(out_dir, 'nsd_clipvision_test.npy'), test_clip)

        for i, img_batch in enumerate(tqdm(train_loader, desc="Processing Train Set")):
            img_batch = img_batch.to(device)
            clip_features = net.clip_encode_vision(img_batch)
            train_clip[i] = clip_features[0].cpu().numpy()

        np.save(os.path.join(out_dir, 'nsd_clipvision_train.npy'), train_clip)

    print(" CLIP-Vision feature extraction complete.")

# ----------------------- Main Execution --------------------------
def main():
    parser = argparse.ArgumentParser(description='CLIP-Vision Feature Extractor')
    parser.add_argument("-sub", "--sub", help="Subject Number", default=1, type=int)
    args = parser.parse_args()
    sub = int(args.sub)
    assert sub in [1, 2, 5, 7]

    # Load model
    print("Loading Model:")
    cfgm_name = 'vd_noema'
    pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
    cfgm = model_cfg_bank()(cfgm_name)
    net = get_model()(cfgm)
    state_dict = torch.load(pth, map_location='cpu')
    net.load_state_dict(state_dict, strict=False)

    # Move CLIP module to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.clip = net.clip.to(device)

    # Run feature extraction
    extract_clip_vision_features(sub=sub, net=net, device=device)

if __name__ == "__main__":
    main()
