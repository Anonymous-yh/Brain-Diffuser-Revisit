"""
This script performs ridge regression to map preprocessed fMRI signals to CLIP-Text features 
for each word token. It normalizes fMRI data, trains a separate model for each token embedding, 
predicts test-time embeddings, and saves both the predictions and regression weights.

running: python reconstruction/cliptext_regression.py -sub x
"""

import sys
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse
import os

def load_fmri_data(sub):
    train_path = f'data/processed_data/subj{sub:02d}/nsd_train_fmriavg_nsdgeneral_sub{sub}.npy'
    test_path = f'data/processed_data/subj{sub:02d}/nsd_test_fmriavg_nsdgeneral_sub{sub}.npy'
    train_fmri = np.load(train_path) / 300
    test_fmri = np.load(test_path) / 300
    return train_fmri, test_fmri

def normalize_fmri(train_fmri, test_fmri):
    norm_mean = np.mean(train_fmri, axis=0)
    norm_std = np.std(train_fmri, axis=0, ddof=1)
    train_norm = (train_fmri - norm_mean) / norm_std
    test_norm = (test_fmri - norm_mean) / norm_std
    return train_norm, test_norm

def load_clip_features(sub):
    train_clip = np.load(f'data/extracted_features/subj{sub:02d}/nsd_cliptext_train.npy')
    test_clip = np.load(f'data/extracted_features/subj{sub:02d}/nsd_cliptext_test.npy')
    return train_clip, test_clip

def run_ridge_regression(train_fmri, test_fmri, train_clip, test_clip, alpha=1e5, max_iter=50000):
    num_samples, num_embed, num_dim = train_clip.shape
    num_voxels = train_fmri.shape[1]

    reg_w = np.zeros((num_embed, num_dim, num_voxels), dtype=np.float32)
    reg_b = np.zeros((num_embed, num_dim), dtype=np.float32)
    pred_clip = np.zeros_like(test_clip)

    print("Training Regression...")
    for i in range(num_embed):
        reg = skl.Ridge(alpha=alpha, max_iter=max_iter, fit_intercept=True)
        reg.fit(train_fmri, train_clip[:, i])
        reg_w[i] = reg.coef_
        reg_b[i] = reg.intercept_

        pred_test = reg.predict(test_fmri)
        pred_std = (pred_test - np.mean(pred_test, axis=0)) / np.std(pred_test, axis=0)
        pred_clip[:, i] = pred_std * np.std(train_clip[:, i], axis=0) + np.mean(train_clip[:, i], axis=0)

        print(f"Embedding {i:3d} | Test R²: {reg.score(test_fmri, test_clip[:, i]):.4f}")

    return reg_w, reg_b, pred_clip

def save_outputs(sub, reg_w, reg_b, pred_clip):
    os.makedirs(f'data/predicted_features/subj{sub:02d}', exist_ok=True)
    os.makedirs(f'data/regression_weights/subj{sub:02d}', exist_ok=True)

    np.save(f'data/predicted_features/subj{sub:02d}/nsd_cliptext_predtest_nsdgeneral.npy', pred_clip)
    
    with open(f'data/regression_weights/subj{sub:02d}/cliptext_regression_weights.pkl', "wb") as f:
        pickle.dump({'weight': reg_w, 'bias': reg_b}, f)

def main():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument("-sub", "--sub", help="Subject Number", default=1, type=int)
    args = parser.parse_args()
    sub = int(args.sub)
    assert sub in [1, 2, 5, 7]

    train_fmri, test_fmri = load_fmri_data(sub)
    train_fmri, test_fmri = normalize_fmri(train_fmri, test_fmri)
    train_clip, test_clip = load_clip_features(sub)
    reg_w, reg_b, pred_clip = run_ridge_regression(train_fmri, test_fmri, train_clip, test_clip)
    save_outputs(sub, reg_w, reg_b, pred_clip)

if __name__ == "__main__":
    main()
