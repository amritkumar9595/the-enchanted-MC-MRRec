"""
the reconstructions of the validation data is compared wrt to the GT.
The corresponding metrics are obtained

"""


import numpy as np
import h5py
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from sewar.full_ref import vifp

import logging
import pathlib
import random
import shutil
import time
import h5py
import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common.args import Args
from common.subsample import MaskFunc
from data import transforms as T
from data.data_loader2 import SliceData , DataTransform
from models.models import UnetModel,DataConsistencyLayer , _NetG , network , _NetG_lite , network_dun,SSIM
from tqdm import tqdm
import data.transforms as T



def metrics(rec,ref):

    ssim = np.zeros(rec.shape[0])
    psnr = np.zeros(rec.shape[0])
    vif = np.zeros(rec.shape[0])
    
    for ii in range(rec.shape[0]):
        data_range = np.maximum(ref[ii].max(),rec[ii].max()) - np.minimum(ref[ii].min(),rec[ii].min())
        ssim[ii] = structural_similarity(ref[ii],rec[ii],data_range= data_range)
        psnr[ii] = peak_signal_noise_ratio(ref[ii],rec[ii],data_range= data_range)
        vif[ii] =  vifp(ref[ii],rec[ii],sigma_nsq = 0.4)

    return ssim,psnr,vif

def load_data(args,fname):
    
    f_recons = args.recons_path + fname
    f_gt = args.data_path + fname

    
    with h5py.File(f_recons, 'r') as data:
        img_recons = data['Recons'][()]

    with h5py.File(f_gt, 'r') as data:
            ksp_cmplx = data['kspace'][()]
            img_gt_np = T.zero_filled_reconstruction(ksp_cmplx).unsqueeze(0)    
    img_gt_np =img_gt_np.numpy()
    print("img_recons , img_gt_np",img_recons.shape, img_gt_np.shape)
    return img_recons , img_gt_np
            
    
    



def evaluate(args):

    examples=[]
    files = list(pathlib.Path(args.recons_path).iterdir())
    for fname in sorted(files):
        examples.append(str(fname))
    print("len",len(examples))
    m = np.zeros((len(examples),3))
    

    for i in range(len(examples)):
        print(i)
        fname = examples[i]
        fname = (pathlib.Path(str(fname)))
        parts = list(fname.parts)
        
        file = parts[-1]
        
        

        rec , ref = load_data(args,file)   ## divied by max value to be included!
        rec = rec/rec.max()
        ref = ref/ref.max()
        
        print("max_rec_ref",rec.max(),ref.max())
                
        ssim,psnr,vif = metrics(rec,ref)
        
        
        m[i,0] = ssim
        m[i,1] = psnr
        m[i,2] = vif

    print("m",m)
    ssim = m[:,0].mean()
    psnr = m[:,1].mean()
    vif = m[:,2].mean()
    
    print("ssim,psnr,vif",ssim,psnr,vif)

def main(args):
    
    evaluate(args)
    

    
    
    
    
def create_arg_parser():
    parser = Args()

    parser.add_argument('--data-path', type=str,help='Path to the dataset')
    parser.add_argument('--recons-path', type=str, help='Path where reconstructions are to be saved')
    parser.add_argument('--acceleration', type=int,help='Ratio of k-space columns to be sampled. 5x or 10x masks provided')

    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
  
    main(args)