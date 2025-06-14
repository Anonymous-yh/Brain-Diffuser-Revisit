"""
This file use pretrained VD-VAE to extract latent variables
The latent variables are used in training process(vae_regression.py)

running : python reconstruction/vae_extract_feature.py -sub x
"""

import sys
sys.path.append('vdvae')
import torch
import numpy as np
#from mpi4py import MPI
import socket
import argparse
import os
import json
import subprocess
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import (logger,
                   local_mpi_rank,
                   mpi_size,
                   maybe_download,
                   mpi_rank)
from data import mkdir_p
from contextlib import contextmanager
import torch.distributed as dist
#from apex.optimizers import FusedAdam as AdamW
from vae import VAE
from torch.nn.parallel.distributed import DistributedDataParallel
from train_helpers import restore_params
from image_utils import *
from model_utils import *
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import pickle
from tqdm import tqdm
import argparse

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='VD-VAE Feature Extraction')
    parser.add_argument("-sub", "--sub", help="Subject Number", default=1, type=int)
    parser.add_argument("-bs", "--bs", help="Batch Size", default=30, type=int)
    return parser.parse_args()

# Dataset class
class ExternalImageGenerator(Dataset):
    def __init__(self, data_path):
        self.im = np.load(data_path).astype(np.uint8)

    def __getitem__(self, index):
        img = Image.fromarray(self.im[index])
        img = T.functional.resize(img, (64, 64))
        img = torch.tensor(np.array(img)).float()
        return img

    def __len__(self):
        return len(self.im)

# Dotdict for config
class dotdict(dict):
    """dot.notation access to dict attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def extract_latents(dataloader, vae_model, preprocess_fn, num_latents, batch_size):
    all_latents = []
    for batch_idx, batch_imgs in enumerate(tqdm(dataloader, desc="Extracting features")):
        data_input, _ = preprocess_fn(batch_imgs)
        with torch.no_grad():
            activations = vae_model.encoder.forward(data_input)
            _, stats = vae_model.decoder.forward(activations, get_latents=True)
            batch_latent = [stats[i]['z'].cpu().numpy().reshape((len(data_input), -1)) for i in range(num_latents)]
            all_latents.append(np.hstack(batch_latent))
    return np.concatenate(all_latents)

def main():
    args = parse_args()
    sub = args.sub
    batch_size = args.bs
    assert sub in [1, 2, 5, 7], "Subject must be one of [1,2,5,7]"

    # Hyperparameters / config dictionary
    H = {'image_size': 64, 'image_channels': 3,'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 
     'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th', 
     'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th', 'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 
     'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999, 
     'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 
     'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 
     'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 
     'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 
     'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 
     'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None, 
     'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 
     'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}
    H = dotdict(H)

    # Setup preprocessing function and load VAE
    H, preprocess_fn = set_up_data(H)
    ema_vae = load_vaes(H)

    # Load datasets
    train_img_path = f'data/processed_data/subj{sub:02d}/nsd_train_stim_sub{sub}.npy'
    test_img_path = f'data/processed_data/subj{sub:02d}/nsd_test_stim_sub{sub}.npy'

    train_dataset = ExternalImageGenerator(train_img_path)
    test_dataset = ExternalImageGenerator(test_img_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Extract latents from train and test datasets
    num_latents = 31
    print("Extracting training latents...")
    train_latents = extract_latents(train_loader, ema_vae, preprocess_fn, num_latents, batch_size)
    print("Extracting testing latents...")
    test_latents = extract_latents(test_loader, ema_vae, preprocess_fn, num_latents, batch_size)

    # Save to npz file
    save_dir = f"data/extracted_features/subj{sub:02d}"
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, "nsd_vdvae_features_31l.npz"),
        train_latents=train_latents,
        test_latents=test_latents
    )
    print(f"Saved features to {save_dir}")

if __name__ == '__main__':
    main()