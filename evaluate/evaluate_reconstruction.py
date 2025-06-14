"""
This script evaluates reconstructed images against ground-truth using feature-based (pairwise correlation or distance) 
and pixel-based (SSIM, pixel-wise correlation) metrics across multiple pretrained vision models.

running: python evaluate/evaluate_reconstruction.py -sub x -method 'brain-diffuser'
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import binom
from scipy.spatial.distance import correlation
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import seaborn as sns


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument("-sub", "--sub", help="Subject Number", type=int, default=1)
    parser.add_argument("-method", "--method", help="Generate Method", default='brain-diffuser')
    return parser.parse_args()

def pairwise_corr_all(gt, pred):
    r = np.corrcoef(gt, pred)
    r = r[:len(gt), len(gt):]
    congruent = np.diag(r)
    success = r < congruent[:, None]
    success_cnt = np.sum(success, axis=1)
    perf = np.mean(success_cnt) / (len(gt) - 1)
    p_val = 1 - binom.cdf(perf * len(gt) * (len(gt) - 1), len(gt) * (len(gt) - 1), 0.5)
    return perf, p_val

def evaluate_features(sub, method, net_list, num_test=982):
    feats_dir = f'data/eval_features/subj{sub:02d}'
    test_dir = 'data/eval_features/test_images'

    pairwise_corrs = []
    for net_name, layer in net_list:
        gt_path = f'{test_dir}/{net_name}_{layer}.npy'
        eval_path = f'{feats_dir}/{net_name}_{layer}_{method}.npy'

        gt_feat = np.load(gt_path).reshape((num_test, -1))
        eval_feat = np.load(eval_path).reshape((num_test, -1))

        print(f"\n{net_name}, Layer: {layer}")
        if net_name in ['efficientnet', 'swav']:
            dist = np.mean([correlation(gt_feat[i], eval_feat[i]) for i in range(num_test)])
            print(f"Distance: {dist:.4f}")
        else:
            corr, _ = pairwise_corr_all(gt_feat, eval_feat)
            pairwise_corrs.append(corr)
            print(f"Pairwise Corr: {corr:.4f}")
    return pairwise_corrs

def plot_feature_correlations(corrs, net_list):
    names = [f"{net}_{layer}" for net, layer in net_list if net not in ['efficientnet', 'swav']]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=names, y=corrs)
    plt.xticks(rotation=45)
    plt.ylabel("Pairwise Correlation")
    plt.title("Feature-based Correlation per Network")
    plt.tight_layout()
    plt.savefig("feature_correlation_barplot.png")
    plt.show()

def get_generated_image_path(i, sub, method):
    if method == 'brain-diffuser':
        return f'results/versatile_diffusion/subj{sub:02d}/{i}.png'
    elif method == 'only-VDVAE':
        return f'results/vdvae/subj{sub:02d}/{i}.png'
    elif method == 'wo-cliptext':
        return f'results/versatile_diffusion/without_cliptext/subj{sub:02d}/{i}.png'
    elif method == 'wo-clipvision':
        return f'results/versatile_diffusion/without_clipvision/subj{sub:02d}/{i}.png'
    else:
        raise ValueError(f"Unknown method: {method}")

def evaluate_images(sub, method, num_test=982):
    ssim_scores = []
    pixcorr_scores = []

    for i in tqdm(range(num_test), desc="Evaluating Images"):
        gen_path = get_generated_image_path(i, sub, method)
        gt_path = f'data/nsddata_stimuli/test_images/{i}.png'

        gen_img = Image.open(gen_path).resize((425, 425))
        gt_img = Image.open(gt_path)

        gen_arr = np.array(gen_img) / 255.0
        gt_arr = np.array(gt_img) / 255.0

        pixcorr = np.corrcoef(gen_arr.reshape(1, -1), gt_arr.reshape(1, -1))[0, 1]
        pixcorr_scores.append(pixcorr)

        gen_gray = rgb2gray(gen_arr)
        gt_gray = rgb2gray(gt_arr)

        ssim_score = ssim(gen_gray, gt_gray, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
        ssim_scores.append(ssim_score)

    print(f"\nPixel-wise Correlation (Mean): {np.mean(pixcorr_scores):.4f}")
    print(f"SSIM (Mean): {np.mean(ssim_scores):.4f}")

    return pixcorr_scores, ssim_scores

def plot_image_metrics(ssim_list, pixcorr_list):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(ssim_list, kde=True, bins=30, color='blue')
    plt.title("SSIM Distribution")
    plt.xlabel("SSIM")

    plt.subplot(1, 2, 2)
    sns.histplot(pixcorr_list, kde=True, bins=30, color='green')
    plt.title("Pixel-wise Correlation Distribution")
    plt.xlabel("Pixel Correlation")

    plt.tight_layout()
    plt.savefig("image_quality_metrics.png")
    plt.show()


def main():
    args = parse_arguments()
    assert args.sub in [1, 2, 5, 7], "Subject must be in [1, 2, 5, 7]"

    net_list = [
        ('inceptionv3', 'avgpool'),
        ('clip', 'final'),
        ('alexnet', 2),
        ('alexnet', 5),
        ('efficientnet', 'avgpool'),
        ('swav', 'avgpool')
    ]

    pairwise_corrs = evaluate_features(args.sub, args.method, net_list)
    plot_feature_correlations(pairwise_corrs, net_list)
    pixcorr_list, ssim_list = evaluate_images(args.sub, args.method)
    plot_image_metrics(ssim_list, pixcorr_list)

    output_dir = f"figures/subj{args.sub:02d}/{args.method}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/image_quality_metrics.png")

if __name__ == '__main__':
    main()
