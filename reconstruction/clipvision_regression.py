""""
This script trains Ridge Regression models to map preprocessed fMRI signals to CLIP-Vision features. 
Each embedding channel is predicted separately. After normalization and training, the predicted features 
for the test set are saved along with the learned weights and biases.

running: python reconstruction/clipvision_regression.py -sub x
"""

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

def load_clip_vision_features(sub):
    train_clip = np.load(f'data/extracted_features/subj{sub:02d}/nsd_clipvision_train.npy')
    test_clip = np.load(f'data/extracted_features/subj{sub:02d}/nsd_clipvision_test.npy')
    return train_clip, test_clip

def run_ridge_regression(train_fmri, test_fmri, train_clip, test_clip, alpha=60000, max_iter=50000):
    num_samples, num_embed, num_dim = train_clip.shape
    num_voxels = train_fmri.shape[1]

    reg_w = np.zeros((num_embed, num_dim, num_voxels), dtype=np.float32)
    reg_b = np.zeros((num_embed, num_dim), dtype=np.float32)
    pred_clip = np.zeros_like(test_clip)

    print("Training Ridge Regression...")
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

    np.save(f'data/predicted_features/subj{sub:02d}/nsd_clipvision_predtest_nsdgeneral.npy', pred_clip)
    
    with open(f'data/regression_weights/subj{sub:02d}/clipvision_regression_weights.pkl', "wb") as f:
        pickle.dump({'weight': reg_w, 'bias': reg_b}, f)

def main():
    parser = argparse.ArgumentParser(description='Regression from fMRI to CLIP-Vision Features')
    parser.add_argument("-sub", "--sub", help="Subject Number", default=1, type=int)
    args = parser.parse_args()
    sub = args.sub
    assert sub in [1, 2, 5, 7]

    train_fmri, test_fmri = load_fmri_data(sub)
    train_fmri, test_fmri = normalize_fmri(train_fmri, test_fmri)
    train_clip, test_clip = load_clip_vision_features(sub)
    reg_w, reg_b, pred_clip = run_ridge_regression(train_fmri, test_fmri, train_clip, test_clip)
    save_outputs(sub, reg_w, reg_b, pred_clip)

if __name__ == "__main__":
    main()
