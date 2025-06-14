"""
This file finishes the first-stage reconstruction

"""
import sys
sys.path.append('vdvae')
import torch
import numpy as np
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
from PIL import Image
import os

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-bs", "--bs",help="Batch Size",default=30)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]
batch_size=int(args.bs)

# 1.load VAE

# For parameter configurations and hyperparameter settings, please refer to hps.py
H = {'image_size': 64, 'image_channels': 3,'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 
     'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th', 
     'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th', 'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 
     'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999, 
     'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 
     'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 
     'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 
     'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 
     'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 
     'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 
     'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None, 
     'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 
     'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}

# use dot notation to access, modify, and delete elements of a dictionary.
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

H = dotdict(H)
H, preprocess_fn = set_up_data(H)

ema_vae = load_vaes(H)

# To get Latents' shape,we have to encoder test data

class ExternalImageGenerator(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)

    def __getitem__(self, index):
        img = Image.fromarray(self.im[index])
        img = T.functional.resize(img,(64,64))
        img = torch.tensor(np.array(img)).float()
        return img
    
    def __len__(self):
        return len(self.im)

test_img_path = 'data/processed_data/subj{:02d}/nsd_test_stim_sub{}.npy'.format(sub,sub)
test_imgs = ExternalImageGenerator(data_path=test_img_path)
test_loader = DataLoader(test_imgs,batch_size,shuffle=False)
num_latents = 31
test_latents = []

for i, x in enumerate(tqdm(test_loader, desc="Encoding test data")):
    data_input, target = preprocess_fn(x)

    with torch.no_grad():
        activations = ema_vae.encoder.forward(data_input)
        px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)


# The function transfers latents from 
def transfer_latents(latents,ref):
    layer_dims = np.array([2**4,2**4,2**8,2**8,2**8,2**8,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**10,
                           2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**14])
    trans_latents = []
    for i in range(31):
        t_latents = latents[:,layer_dims[:i].sum():layer_dims[:(i+1)].sum()]
        c,h,w = ref[i]['z'].shape[1:]
        trans_latents.append(t_latents.reshape(len(latents),c,h,w))

    return trans_latents


pred_latents = np.load('data/predicted_features/subj{:02d}/nsd_vdvae_nsdgeneral_pred_sub{}_31l_alpha50k.npy'.format(sub,sub))
ref_latents = stats

input_latent = transfer_latents(pred_latents[range(len(test_imgs))],ref_latents)


def sample_from_hier_latents(latents,sample_ids):
  sample_ids = [id for id in sample_ids if id<len(latents[0])]
  layers_num=len(latents)
  sample_latents = []
  for i in range(layers_num):
    sample_latents.append(torch.tensor(latents[i][sample_ids]).float().cuda())
  return sample_latents


for i in tqdm(range(int(np.ceil(len(test_imgs) / batch_size))), desc="Batches"):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    samp = sample_from_hier_latents(input_latent, range(start_idx, end_idx))
    
    # 解码 latent 成图像
    px_z = ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
    sample_from_latent = ema_vae.decoder.out_net.sample(px_z)

    # 内层：保存当前 batch 的图像
    for j in tqdm(range(len(sample_from_latent)), desc=f"Saving images (batch {i})", leave=False):
        im = sample_from_latent[j]
        im = Image.fromarray(im)
        im = im.resize((512, 512), resample=3)
        im.save(f'results/vdvae/subj{sub:02d}/{start_idx + j}.png')


