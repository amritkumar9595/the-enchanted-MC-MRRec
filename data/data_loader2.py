import torch
import pathlib
import random
import data.transforms as T
import h5py
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
from bart import bart
import math
import random

   
class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform,sample_rate,acceleration):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace',
                filename',  'sensitivity maps', and 'acclearation' as inputs. 

            sample_rate : A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
            acceleration: Whether to use 5x US data or 10x US data
        """
        self.acceleration = acceleration

        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(root).iterdir())

        
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
            
        for fname in sorted(files):

                self.examples.append(str(fname)) # [(fname, slice) for slice in range(50,num_slices-50)]

            
    def __len__(self):

        return len(self.examples)
    

    def __getitem__(self, i):
        fname = self.examples[i]
        
        with h5py.File(fname, 'r') as data:

            ksp = data['kspace'][()]
            sens = data['sensitivity'][()]
        

        return self.transform (ksp, fname, sens, self.acceleration)
    
    
    
    
class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self):   
        """
        Not required !
        """
    
    def __call__(self,ksp_cmplx,fname,sensitivity,acceleration):
        """
        Args:
            kspace (numpy.array): Input k-space of the multi-coil data
            fname (str): File name
            sensitivity maps (numpy.array): ENLIVE sensitivity maps
            acceleartion: whether to train for 5x US ksp or 10x US kspace

        """

        sens_t = T.to_tensor(sensitivity)
        
        ksp_t = T.to_tensor(ksp_cmplx)
        ksp_t = ksp_t.permute(2,0,1,3)
        img_gt = T.ifft2(ksp_t)
        img_gt_sens = T.combine_all_coils(img_gt , sens_t)
        
        img_gt_np = T.zero_filled_reconstruction(ksp_cmplx)
        
        
        if acceleration == 5:

            if ksp_t.shape[2]==170:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x170.npy")
            elif ksp_t.shape[2]==174:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x174.npy")
            elif ksp_t.shape[2]==180:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R5_218x180.npy")
        
        elif acceleration == 10:

            if ksp_t.shape[2]==170:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x170.npy")
            elif ksp_t.shape[2]==174:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x174.npy")
            elif ksp_t.shape[2]==180:
                sp_r5 = np.load("/media/student1/NewVolume/MR_Reconstruction/midl/MC-MRRec-challenge/Data/poisson_sampling/R10_218x180.npy")
        
        

        randint = random.randint(0,99)                   #to get a random mask everytime ! 
        mask = sp_r5[randint]
        mask = torch.from_numpy(mask)
        mask = (torch.stack((mask,mask),dim=-1)).float()
        
        ksp_us = torch.where(mask == 0, torch.Tensor([0]), ksp_t)
 
        img_us = T.ifft2(ksp_us)
        img_us_sens = T.combine_all_coils(img_us , sens_t)
        
        ksp_us_np = ksp_us.numpy()
        ksp_us_cmplx = ksp_us_np[:,:,:,0] + 1j*ksp_us_np[:,:,:,1]
        ksp_us_cmplx = ksp_us_cmplx.transpose(1,2,0)
        
        img_us_np = T.zero_filled_reconstruction(ksp_us_cmplx)
        
        pha_gt = T.phase(img_gt_sens)
        pha_us = T.phase(img_us_sens) 
        
        pha_gt = pha_gt + 3.1415927410125732
        pha_us = pha_us + 3.1415927410125732
        
        mag_gt = T.complex_abs(img_gt_sens)
        mag_us = T.complex_abs(img_us_sens)
        
        mag_gt_pad = T.pad(mag_gt,[256,256] )
        mag_us_pad = T.pad(mag_us,[256,256] )
        
        pha_gt_pad = T.pad(pha_gt,[256,256] )
        pha_us_pad = T.pad(pha_us,[256,256] )
        

        
        return mag_us_pad/mag_us_pad.max() , mag_gt_pad/mag_us_pad.max() , pha_us_pad , pha_gt_pad , ksp_us/mag_us_pad.max() ,img_us_sens/mag_us_pad.max(), img_gt_sens/mag_us_pad.max() , img_us_np/img_us_np.max() , img_gt_np/img_us_np.max() , sens_t , mask ,img_us_np.max(),fname
        
        
        
        
        
        
        
        
        
        
        

        



        
    