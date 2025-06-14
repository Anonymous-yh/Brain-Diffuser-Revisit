"""
This pipeline enables decoding brain activity into images by leveraging predicted latent features 
in a diffusion-based generative framework.

running: python reconstruction/VD_reconstruction.py -sub x
"""

import sys
sys.path.append('versatile_diffusion')
import os
import numpy as np
import torch
import PIL
from PIL import Image
import torchvision.transforms as tvtrans
import argparse

from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='Versatile Diffusion Image Reconstruction')
    parser.add_argument("-sub", "--sub", help="Subject Number", default=1, type=int)
    parser.add_argument("-diff_str", "--diff_str", help="Diffusion Strength", default=0.75, type=float)
    parser.add_argument("-mix_str", "--mix_str", help="Mixing Strength", default=0.4, type=float)
    return parser.parse_args()


def regularize_image(x):
    BICUBIC = PIL.Image.Resampling.BICUBIC
    if isinstance(x, str):
        x = Image.open(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, PIL.Image.Image):
        x = x.resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, np.ndarray):
        x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, torch.Tensor):
        pass
    else:
        raise ValueError('Unknown image type')
    assert (x.shape[1] == 512) & (x.shape[2] == 512), 'Wrong image size'
    return x


def initialize_model():
    cfgm = model_cfg_bank()('vd_noema')
    net = get_model()(cfgm)
    state_dict = torch.load('versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth', map_location='cpu')
    net.load_state_dict(state_dict, strict=False)
    return net, DDIMSampler_VD(net)


def load_latents(sub):
    pred_text = np.load(f'data/predicted_features/subj{sub:02d}/nsd_cliptext_predtest_nsdgeneral.npy')
    pred_vision = np.load(f'data/predicted_features/subj{sub:02d}/nsd_clipvision_predtest_nsdgeneral.npy')
    return torch.tensor(pred_text).half().cuda(1), torch.tensor(pred_vision).half().cuda(1)


def run_inference(sub, strength, mixing, net, sampler, pred_text, pred_vision):
    n_samples = 1
    ddim_steps = 50
    ddim_eta = 0
    scale = 7.5

    net.clip.cuda(0)
    net.autokl.cuda(0).half()

    torch.manual_seed(0)
    for im_id in tqdm(range(len(pred_vision)), desc="Generating Images"):
        image_path = f'results/vdvae/subj{sub:02d}/{im_id}.png'
        image = regularize_image(Image.open(image_path))
        zin = (image * 2 - 1).unsqueeze(0).cuda(0).half()

        init_latent = net.autokl_encode(zin)
        sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

        t_enc = int(strength * ddim_steps)
        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to('cuda:0')).cuda(1)

        utx = net.clip_encode_text('').cuda(1).half()
        uim = net.clip_encode_vision(torch.zeros((1, 3, 224, 224)).cuda(0)).cuda(1).half()

        cim = pred_vision[im_id].unsqueeze(0)
        ctx = pred_text[im_id].unsqueeze(0)

        sampler.model.model.diffusion_model.device = 'cuda:1'
        sampler.model.model.diffusion_model.half().cuda(1)

        h, w = 512, 512
        z = sampler.decode_dc(
            x_latent=z_enc,
            first_conditioning=[uim, cim],
            second_conditioning=[utx, ctx],
            t_start=t_enc,
            unconditional_guidance_scale=scale,
            xtype='image',
            first_ctype='vision',
            second_ctype='prompt',
            mixed_ratio=(1 - mixing),
        )

        z = z.cuda(0).half()
        x = net.autokl_decode(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = [tvtrans.ToPILImage()(xi) for xi in x]

        output_path = f'results/versatile_diffusion/subj{sub:02d}/{im_id}.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        x[0].save(output_path)


def main():
    args = parse_arguments()
    sub = args.sub
    assert sub in [1, 2, 5, 7]

    net, sampler = initialize_model()
    pred_text, pred_vision = load_latents(sub)
    run_inference(sub, args.diff_str, args.mix_str, net, sampler, pred_text, pred_vision)


if __name__ == '__main__':
    main()
