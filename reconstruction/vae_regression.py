"""
In this file,we trained a ridge regression model between fMRI training patterns and the 
concatenated vectors (oncatenated latent variables).
Then we use the ridge regression model to obtain predicted latent variables.

running: python reconstruction/vae_regression.py -sub x
"""
import sys
import numpy as np
import sklearn.linear_model as skl
import argparse
import pickle
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

nsd_feature = np.load('data/extracted_features/subj{:02d}/nsd_vdvae_features_31l.npz'.format(sub))
train_latents = nsd_feature['train_latents']
test_latents = nsd_feature['test_latents']

train_fmri_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
test_fmri_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
train_fmri = np.load(train_fmri_path)
test_fmri = np.load(test_fmri_path)

# 1. preprocessing FMRI
train_fmri = train_fmri/300
test_fmri = test_fmri/300

mean = np.mean(train_fmri,axis=0)
std = np.std(train_fmri,axis=0, ddof=1)
train_fmri = (train_fmri-mean)/std
test_fmri = (test_fmri-mean)/std

fmri_dim , num_train , num_test = train_fmri.shape[1],train_fmri.shape[0],test_fmri.shape[0]

# 2. training ridge regression model

reg = skl.Ridge(alpha=50000, max_iter=10000, fit_intercept=True)
reg.fit(train_fmri, train_latents)
pred_test_latent = reg.predict(test_fmri)
pred_test_latent = (pred_test_latent-np.mean(pred_test_latent,axis=0))/np.std(pred_test_latent,axis=0,ddof=1)
pred_latents = pred_test_latent * np.std(train_latents,axis=0,ddof=1) + np.mean(train_latents,axis=0)

np.save('data/predicted_features/subj{:02d}/nsd_vdvae_nsdgeneral_pred_sub{}_31l_alpha50k.npy'.format(sub,sub),pred_latents)

# save weights 
datadict = {
    'weight' : reg.coef_,
    'bias' : reg.intercept_,

}

with open('data/regression_weights/subj{:02d}/vdvae_regression_weights.pkl'.format(sub),"wb") as f:
  pickle.dump(datadict,f)

