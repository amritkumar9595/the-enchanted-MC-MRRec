"""
The absolute value of reconstructions from the unet_vs net are passed through Down-Up-Network. 
The Down-Up-Network is trained using SSIM loss, with the RSS of FS images as GT.
"""

import logging
import pathlib
import random
import shutil
import time

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
    
    train_data = SliceData(
        root=str(args.data_path) + '/Train',
        transform=DataTransform(),
        sample_rate=args.sample_rate,
        acceleration=args.acceleration
    )
    dev_data = SliceData(
        root=str(args.data_path) + '/Val',
        transform=DataTransform(),
        sample_rate=args.sample_rate,
        acceleration=args.acceleration
    )
    return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)

    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=1,
        shuffle = True,
        num_workers=0,
        pin_memory=False,
    )
    
    return train_loader, dev_loader , display_loader

para =0.1
para = torch.nn.Parameter(torch.Tensor([para]))
ssim_loss = SSIM().cuda()

def train_epoch(args, epoch, model_mag,model_pha , model_vs, model_dun , data_loader, optimizer_dun, writer):

    model_mag.eval()
    model_pha.eval()
    model_vs.eval()
    model_dun.eval()
    
    avg_loss_dun = 0.
    
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    
    for iter, data in enumerate(tqdm(data_loader)):
       
        mag_us,mag_gt,pha_us,pha_gt,ksp_us ,img_us,img_gt,img_us_np,img_gt_np,sens,mask,_,_ = data

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
        inp_dun = T.inp_to_dun(out_cmplx).float().to(args.device)        ## takes only mag
        
        out_dun = model_dun(inp_dun)
        
        # loss_dun = F.mse_loss(out_dun , img_gt_np)
        loss_dun =  ssim_loss(out_dun, img_gt_np,torch.tensor(1.0).unsqueeze(0).cuda())
                    
        optimizer_dun.zero_grad()
    
        loss_dun.backward()
    
        optimizer_dun.step()

        avg_loss_dun = 0.99 * avg_loss_dun + 0.01 * loss_dun.item() if iter > 0 else loss_dun.item()

        writer.add_scalar('TrainLoss_cmplx', loss_dun.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
            f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
            f'Iter = [{iter:4d}/{len(data_loader):4d}] '
            f'Loss_dun = {loss_dun.item():.4g} Avg Loss dun = {avg_loss_dun:.4g} '
            f'Time = {time.perf_counter() - start_iter:.4f}s',
        )
        start_iter = time.perf_counter()
            
            
            
    return avg_loss_dun,  time.perf_counter() - start_epoch


def evaluate(args, epoch, model_mag , model_pha , model_vs , model_dun ,  data_loader, writer):

    model_mag.eval()
    model_pha.eval()
    model_vs.eval()
    model_dun.eval()

    losses_dun = []
    start = time.perf_counter()
    with torch.no_grad():
        print(':::::::::::::::: IN EVALUATE :::::::::::::::::::::')
        for iter, data in enumerate(tqdm(data_loader)):
            
            mag_us,mag_gt,pha_us,pha_gt,ksp_us ,img_us,img_gt,img_us_np,img_gt_np,sens,mask,_,_ = data

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
            
            out_dun = model_dun(inp_dun)
            
            # loss_dun = F.mse_loss(out_dun , img_gt_np)
            loss_dun =  ssim_loss(out_dun, img_gt_np,torch.tensor(1.0).unsqueeze(0).cuda())
            losses_dun.append(loss_dun.item())
                
        writer.add_scalar('Dev_Loss_cmplx',np.mean(losses_dun), epoch)
                
        return np.mean(losses_dun) , time.perf_counter() - start


def visualize(args, epoch, model_mag , model_pha, model_vs ,model_dun,  data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    
    model_mag.eval()
    model_pha.eval()
    model_vs.eval()
    model_dun.eval()
    
    with torch.no_grad():
        print(':::::::::::::::: IN VISUALIZE :::::::::::::::::::::')
        for iter, data in enumerate(data_loader):
            
            mag_us,mag_gt,pha_us,pha_gt,ksp_us ,img_us,img_gt,img_us_np,img_gt_np,sens,mask,_,_ = data

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
            inp_dun = T.inp_to_dun(out_cmplx).float().to(args.device)        ## takes only mag 
            
            out_dun = model_dun(inp_dun)
            
            save_image(inp_dun.float(), 'Input')
                
                
            err_dun = torch.abs(out_dun - img_gt_np)
            
            save_image(err_dun,'Error')                           
            save_image(out_dun, 'Recons')
            save_image(img_gt_np, 'Target')
            
            break


def save_model(args, exp_dir, epoch,model_mag,model_pha,model_vs, model_dun, optimizer_dun , best_dev_loss_dun, is_new_best_dun):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'optimizer_dun': optimizer_dun.state_dict(),
            'model_mag' : model_mag.state_dict(),
            'model_pha' : model_pha.state_dict(),
            'model_vs' : model_vs.state_dict(),
            'model_dun':model_dun.state_dict(),
            'best_dev_loss_dun': best_dev_loss_dun,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'dun_model.pt'
    )
    if is_new_best_dun:   
        shutil.copyfile(exp_dir / 'dun_model.pt', exp_dir / 'best_dun_model.pt')

def build_unet(args):

    model1 = UnetModel(
       in_chans=1,
       out_chans=1,
       chans=args.num_chans,
       num_pool_layers=args.num_pools,
       drop_prob=args.drop_prob,
       residual= args.residual
    ).to(args.device)
    
    model2 = UnetModel(
       in_chans=1,
       out_chans=1,
       chans=args.num_chans,
       num_pool_layers=args.num_pools,
       drop_prob=args.drop_prob,
       residual= args.residual
    ).to(args.device)

    return model1, model2



def build_model(args):

    unet_model1 , unet_model2 = build_unet(args)
     
    wacoeff = 0.1
    dccoeff = 0.1
    cascade = 5
    
    model_vs = network(dccoeff, wacoeff, cascade).to(args.device) 
    return unet_model1 , unet_model2 , model_vs



def load_model_pretrained(checkpoint_file):
    
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model_mag , model_pha , model_vs  = build_model(args)
    
    if args.data_parallel:
        model_mag = torch.nn.DataParallel(model_mag)
        model_pha = torch.nn.DataParallel(model_pha)
        model_vs = torch.nn.DataParallel(model_vs)


    model_mag.load_state_dict(checkpoint['model_mag'])
    model_pha.load_state_dict(checkpoint['model_pha'])
    model_vs.load_state_dict(checkpoint['model_vs'])

    
    return model_mag , model_pha , model_vs    

def load_model(checkpoint_file):
    
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model_mag , model_pha , model_vs = build_model(args)
    model_dun =  _NetG().to(args.device)
    
    
    if args.data_parallel:

        model_mag_pha = torch.nn.DataParallel(model_mag_pha)
        model_dun = torch.nn.DataParallel(model_dun)
        
    model_mag.load_state_dict(checkpoint['model_mag'])
    model_pha.load_state_dict(checkpoint['model_pha'])
    model_vs.load_state_dict(checkpoint['model_vs'])   
    model_dun.load_state_dict(checkpoint['model_dun'])

    optimizer_dun = build_optim(args, model_dun.parameters())

    optimizer_dun.load_state_dict(checkpoint['optimizer_dun'])
    
    
    return checkpoint, model_mag, model_pha , model_vs , model_dun , optimizer_dun


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'dun_summary')

    if args.resume == 'True':

        checkpoint, model_mag, model_pha, model_vs ,model_dun , optimizer_dun  = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss_dun = checkpoint['best_dev_loss_dun']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:

        checkpoint_pretrained = args.pretrained
        model_mag , model_pha , model_vs = load_model_pretrained(checkpoint_pretrained)
        model_dun =  _NetG().to(args.device)
        
        if args.data_parallel:

            model_dun = torch.nn.DataParallel(model_dun)

        optimizer_dun = build_optim(args, model_dun.parameters())

        best_dev_loss_dun = 1e9
        start_epoch = 0
        
    logging.info(args)
    logging.info(model_mag)
    logging.info(model_pha)
    logging.info(model_vs)
    logging.info(model_dun)

    train_loader, dev_loader , display_loader = create_data_loaders(args)  
    scheduler_dun = torch.optim.lr_scheduler.StepLR(optimizer_dun, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):

        scheduler_dun.step(epoch)

        train_loss_dun , train_time = train_epoch(args, epoch, model_mag , model_pha , model_vs ,model_dun, train_loader, optimizer_dun , writer)
        dev_loss_dun , dev_time = evaluate(args, epoch, model_mag , model_pha , model_vs ,  model_dun ,  dev_loader, writer)
        visualize(args, epoch, model_mag, model_pha , model_vs , model_dun , display_loader, writer)


        is_new_best_dun = dev_loss_dun < best_dev_loss_dun

        best_dev_loss_dun = min(best_dev_loss_dun, dev_loss_dun)
                
        save_model(args, args.exp_dir, epoch,model_mag , model_pha , model_vs, model_dun, optimizer_dun, best_dev_loss_dun , is_new_best_dun)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}]  TrainLoss_dun = {train_loss_dun:.4g}  '
            f'DevLoss_dun = {dev_loss_dun:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():
    parser = Args()
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--report-interval', type=int, default=5000, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', type=str, default='False',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--data-path', type=pathlib.Path, default='/media/student1/RemovableVolume/calgary/',
                          help='Path to the dataset')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing Falsecheckpoint. Used along with "--resume"')
    parser.add_argument('--pretrained', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')

    parser.add_argument('--residual', type=str, default='False')
    parser.add_argument('--acceleration', type=int,help='Ratio of k-space columns to be sampled. 5x or 10x masks provided')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
