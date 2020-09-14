"""
the valid.py script saves the reconstruction of the validation data.
these reconstructions are used to calculate the metrices and evaluate model performance

"""

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_datasets(args):

    dev_data = SliceData(
        root=str(args.data_path) + '/Val',
        transform=DataTransform(),
        sample_rate=args.sample_rate,
        acceleration=args.acceleration
    )
    return dev_data

def create_data_loaders(args):
    dev_data = create_datasets(args)

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
    )

    
    return dev_loader 

def reconstruct(args, model_mag , model_pha , model_vs , model_dun ,  data_loader):
    
    model_mag.eval()
    model_pha.eval()
    model_vs.eval()
    model_dun.eval()

    with torch.no_grad():
        
        for iter, data in enumerate(tqdm(data_loader)):
            
            mag_us,mag_gt,pha_us,pha_gt,ksp_us ,img_us,img_gt,img_us_np,img_gt_np,sens,mask,fname = data
            

            inp_mag = mag_us.unsqueeze(1).to(args.device)
            inp_pha = pha_us.unsqueeze(1).to(args.device)
            img_us_np = img_us_np.unsqueeze(1).float().to(args.device)
            img_gt_np = img_gt_np.unsqueeze(1).float().to(args.device)
            ksp_us = ksp_us.to(args.device)
            sens = sens.to(args.device)
            mask = mask.to(args.device)
            img_gt = img_gt.to(args.device)
                                
            out_mag = model_mag(inp_mag)
            out_pha = model_pha(inp_pha)
            
            out_mag_unpad = T.unpad(out_mag , ksp_us)
            out_pha_unpad = T.unpad(out_pha , ksp_us) 
            out_cmplx = T.dc(out_mag,out_pha,ksp_us,sens,mask)
            out_cmplx = model_vs(out_cmplx,ksp_us,sens,mask)
            
            # inp_dun = T.rss(out_cmplx,sens).float().to(args.device)                 ## takes rss
            inp_dun = T.inp_to_dun(out_cmplx).float().to(args.device)          ## takes only mag
            
            out_dun = model_dun(inp_dun).squeeze(0)
            out_dun = out_dun.cpu()
                       
            fname = (pathlib.Path(str(fname)))
            parts = list(fname.parts)
            parts[-2] = 'Recons/acc_' + str(args.acceleration) + 'x'
            path=pathlib.Path(*parts) 
            path = str(path)[2:-2]
            
            with h5py.File(str(path), 'w') as f:
                f.create_dataset("Recons", data=out_dun)

        
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


def main(args):
    
    dev_loader  = create_data_loaders(args) 
    model_mag , model_pha , model_vs , model_dun = load_model(args.model_path)
    reconstruct(args, model_mag , model_pha , model_vs , model_dun ,  dev_loader)
    
    
def create_arg_parser():
    parser = Args()
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=1, type=int, help='Mini batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--model-path', type=str, help='Path where model is present')
    parser.add_argument('--data-path', type=pathlib.Path,help='Path to the dataset')
    parser.add_argument('--recons-path', type=str, help='Path where reconstructions are to be saved')
    parser.add_argument('--acceleration', type=int,help='Ratio of k-space columns to be sampled. 5x or 10x masks provided')

    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
  
    main(args)