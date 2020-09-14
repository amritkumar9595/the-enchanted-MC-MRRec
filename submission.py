"""
the US images of the test set are passed through the trained network and saved with the corresponding file names.
the location of the test dataset and the corresponding trained model are passed as argument

"""

import h5py
import numpy as np
import data.transforms as T
from matplotlib import pyplot as plt
import torch
import math
from torch.nn import functional as F
from collections import namedtuple
from pathlib import Path
from torch import nn
import cv2 as  cv
import torch.nn as nn
import pathlib
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
from models.models import UnetModel,DataConsistencyLayer , _NetG , network , _NetG_lite , network_dun,SSIM
import random
import os

import numpy as np


class SliceData(Dataset):
    
    def __init__(self,transform,root):

        self.transform = transform
        # self.root = root
        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        
        sample_rate = 1.0    ## no need of sampling, remove it later !!
        if sample_rate < 1:
            random.shuffle(files)
        num_files = round(len(files) * sample_rate)
        files = files[:num_files]
        
        
        
        for fname in sorted(files):

                num_slices = 256
                self.examples += [(fname, slice) for slice in range(50,num_slices-50)]

            
    def __len__(self):

        return len(self.examples)
    

    def __getitem__(self, i):
        data_path, slice = self.examples[i]
        sens_path = str(data_path)[:43] + '/sens/' + str(data_path)[49:]
        print("data_path",data_path)
        print("sens_path",sens_path)
        print("slice",slice)
        with h5py.File(data_path, 'r') as data:

            zf_kspace = data['kspace'][()]
            mask_sampling = ~( np.abs(zf_kspace).sum( axis = (0, -1) ) == 0)
            mask_np = 1.0*mask_sampling
            ksp = zf_kspace[slice]
            
        with open(sens_path,'rb') as f:
            sensitivity = np.load(f)
            sensitivity = sensitivity[slice]
        
        return self.transform (ksp, sensitivity, mask_np,data_path,slice)
    

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self):
        """
        We will, we will rock you !!
        
        """    
    def __call__(self,ksp,sens,mask,fname,slice):
        
        mask = torch.from_numpy(mask)
        mask = (torch.stack((mask,mask),dim=-1)).float()
        
        ksp_cmplx = ksp[:,:,::2] + 1j*ksp[:,:,1::2]
        sens_t = T.to_tensor(sens)
        ksp_t = T.to_tensor(ksp_cmplx)
        ksp_us= ksp_t.permute(2,0,1,3)
        
        img_us = T.ifft2(ksp_us)
        img_us_sens = T.combine_all_coils(img_us , sens_t)
        
        pha_us = T.phase(img_us_sens)
        mag_us = T.complex_abs(img_us_sens)
                                                  
        mag_us_pad = T.pad(mag_us,[256,256] )
        pha_us_pad = T.pad(pha_us,[256,256] )
        
        ksp_us_np = ksp
        ksp_us_np = ksp_us_np[:,:,::2] + 1j*ksp_us_np[:,:,1::2]
        

        
        img_us_np = T.zero_filled_reconstruction(ksp_us_np)


        return mag_us_pad/mag_us_pad.max() , pha_us_pad , ksp_us/mag_us_pad.max() , sens_t , mask, fname.name, slice,img_us_np.max()  
    
def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
   
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)    





def create_data_loaders(root):
    
    dev_data = SliceData(transform=DataTransform(),root = root)
    
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    
    return dev_loader

def run_submission(model_mag , model_pha , model_vs , model_dun, data_loader):
    
    model_mag.eval()
    model_pha.eval()
    model_vs.eval()
    model_dun.eval()
    
    reconstructions = defaultdict(list)
    for iter, data in enumerate(tqdm(data_loader)):
        
        mag_us,pha_us,ksp_us,sens,mask,fname, slice, max_mag = data
        inp_mag = mag_us.unsqueeze(1).cuda()
        inp_pha = pha_us.unsqueeze(1).cuda()
        ksp_us = ksp_us.cuda()
        sens = sens.cuda()
        mask = mask.cuda()
        
        out_mag = model_mag(inp_mag)
        out_pha = model_pha(inp_pha)

        out_mag_unpad = T.unpad(out_mag , ksp_us)
        out_pha_unpad = T.unpad(out_pha , ksp_us) 
        out_cmplx = T.dc(out_mag,out_pha,ksp_us,sens,mask)
        out_cmplx = model_vs(out_cmplx,ksp_us,sens,mask)

        # inp_dun = T.rss(out_cmplx,sens).float().to(args.device)                 ## takes rss
        inp_dun = T.inp_to_dun(out_cmplx).float().cuda()          ## takes only mag

        out_dun = model_dun(inp_dun)#.squeeze(0).squeeze(0)
        out_dun = T.unpad(out_dun , ksp_us)
        
        out_dun = out_dun.detach().cpu()
        out_dun = out_dun.squeeze(1)#.squeeze(0)
        out_dun = out_dun*max_mag.float()
        # print("out_dun",out_dun.shape)

        for i in range(1):
            #                 recons[i] = recons[i] * std[i] + mean[i]
            reconstructions[fname[i]].append((slice[i].numpy(), out_dun[i].numpy()))
            
        
    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
        }
    return reconstructions


def build_model(args):
    print("device",args.device)
    model_mag = UnetModel(
       in_chans=1,
       out_chans=1,
       chans=args.num_chans,
       num_pool_layers=args.num_pools,
       drop_prob=args.drop_prob,
       residual= args.residual
    ).to(args.device)
    
    model_pha = UnetModel(
       in_chans=1,
       out_chans=1,
       chans=args.num_chans,
       num_pool_layers=args.num_pools,
       drop_prob=args.drop_prob,
       residual= args.residual
    ).to(args.device)
    
    wacoeff = 0.1
    dccoeff = 0.1
    cascade = 5
    
    model_vs = network(dccoeff, wacoeff, cascade).to(args.device) 
    
    model_dun =  _NetG().to(args.device)
    
    return model_mag , model_pha , model_vs , model_dun


def load_model(checkpoint_file):
    
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    print("check_point file = ",checkpoint_file)
    model_mag , model_pha , model_vs , model_dun = build_model(args)

    model_mag.load_state_dict(checkpoint['model_mag'])
    model_pha.load_state_dict(checkpoint['model_pha'])
    model_vs.load_state_dict(checkpoint['model_vs'])   
    model_dun.load_state_dict(checkpoint['model_dun'])
    
    print("trained model loaded.....")
    
    return model_mag , model_pha , model_vs , model_dun




def main(test_data_path,model_path,out_dir):

    data_loader = create_data_loaders(test_data_path)
    model_mag , model_pha , model_vs , model_dun = load_model(model_path)
    # print(model)
    reconstructions = run_submission(model_mag , model_pha , model_vs , model_dun, data_loader)
    save_reconstructions(reconstructions, Path(out_dir))
    

if __name__ == '__main__':
    
                                            #########  12 channel  #######

    # test_data_path = "/media/student1/RemovableVolume/calgary_new/Test/test_12_channel/Test-R=5/"
    # model_path = "/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/actual/acc_5x/best_dun_model.pt"
    # out_dir = '/media/student1/RemovableVolume/calgary/team_the_enchanted_v3/Track01/12-channel-R=5'
    # out_dir = '/media/student1/RemovableVolume/calgary/Test'
    
    # test_data_path = "/media/student1/RemovableVolume/calgary_new/Test/test_12_channel/Test-R=10/"
    # model_path = "/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/actual/acc_10x/best_dun_model.pt"
    # out_dir = '/media/student1/RemovableVolume/calgary/team_the_enchanted_v3/Track01/12-channel-R=10'
    
                                            #########  32 channel  #######
    
    test_data_path = "/media/student1/RemovableVolume/calgary_new/Test/test_32_channel/Test-R=5/"
    model_path = "/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/actual/acc_5x/best_dun_model.pt"
    out_dir = '/media/student1/RemovableVolume/calgary/team_the_enchanted_v3/Track02/32-channel-R=5'
    
    # test_data_path = "/media/student1/RemovableVolume/calgary_new/Test/test_32_channel/Test-R=10/"
    # model_path = "/media/student1/NewVolume/MR_Reconstruction/experiments/challenge_calgary/actual/acc_10x/best_dun_model.pt"
    # out_dir = '/media/student1/RemovableVolume/calgary/team_the_enchanted_v3/Track02/32-channel-R=10'
        
    
    main(test_data_path,model_path,out_dir)    
    