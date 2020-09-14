"""
step1: Train magnitude and phase network (U-Net) separately.
Step2: Combine magnitude and phase, to obtain the complex image and then train magnitude and phase networks train jointly.
Step3: the model is modified by appending Convnets after the magnitude and phase networks and is trained end to end.

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
from models.models import UnetModel,DataConsistencyLayer , _NetG_lite , network
from tqdm import tqdm



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
        num_workers=2,
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



def train_epoch(args, epoch, model_mag, model_pha,model_vs, data_loader, optimizer_mag, optimizer_pha,optimizer_vs, writer):
    model_mag.train()
    model_pha.train()
    model_vs.train()
    
    avg_loss_mag = 0.
    avg_loss_pha = 0.
    avg_loss_cmplx = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    
    for iter, data in enumerate(tqdm(data_loader)):
       
        mag_us,mag_gt,pha_us,pha_gt,ksp_us ,img_us,img_gt,img_us_np,img_gt_np,sens,mask,_,_ = data

        # input_kspace = input_kspace.to(args.device)
        inp_mag = mag_us.unsqueeze(1).to(args.device)
        tgt_mag = mag_gt.unsqueeze(1).to(args.device)
        inp_pha = pha_us.unsqueeze(1).to(args.device)
        tgt_pha = pha_gt.unsqueeze(1).to(args.device)
        # target = target.unsqueeze(1).to(args.device)
        ksp_us = ksp_us.to(args.device)
        sens = sens.to(args.device)
        mask = mask.to(args.device)
        img_gt = img_gt.to(args.device)

        out_mag = model_mag(inp_mag)
        out_pha = model_pha(inp_pha)
        
        out_mag_unpad = T.unpad(out_mag , ksp_us)
        out_pha_unpad = T.unpad(out_pha , ksp_us)     

        out_cmplx = T.dc(out_mag,out_pha,ksp_us,sens,mask)
        
        
        if (epoch < 1):
            
            model_vs.eval()
        
            loss_mag = F.mse_loss(out_mag, tgt_mag)
            loss_pha = F.mse_loss(out_pha ,tgt_pha)
            loss_cmplx = F.mse_loss(out_cmplx,img_gt.to(args.device))
        
            optimizer_mag.zero_grad()
            optimizer_pha.zero_grad()
        
            loss_mag.backward()
            loss_pha.backward()
        
            optimizer_mag.step()
            optimizer_pha.step()

            avg_loss_mag = 0.99 * avg_loss_mag + 0.01 * loss_mag.item() if iter > 0 else loss_mag.item()
            avg_loss_pha = 0.99 * avg_loss_pha + 0.01 * loss_pha.item() if iter > 0 else loss_pha.item()
            avg_loss_cmplx = 0.99 * avg_loss_cmplx + 0.01 * loss_cmplx.item() if iter > 0 else loss_cmplx.item()
        
            writer.add_scalar('TrainLoss_mag', loss_mag.item(), global_step + iter)
            writer.add_scalar('TrainLoss_pha', loss_pha.item(), global_step + iter)
            writer.add_scalar('TrainLoss_cmplx', loss_cmplx.item(), global_step + iter)
            
            if iter % args.report_interval == 0:
                logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss_mag = {loss_mag.item():.4g} Avg Loss Mag = {avg_loss_mag:.4g} '
                f'Loss_pha = {loss_pha.item():.4g} Avg Loss Pha = {avg_loss_pha:.4g} '
                f'Loss_cmplx = {loss_cmplx.item():.4g} Avg Loss cmplx = {avg_loss_cmplx:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            
            start_iter = time.perf_counter()
            
            
        elif (epoch >= 1 and epoch < 2):
                
            model_mag.train()
            model_pha.train()
            model_vs.eval()
            
            loss_mag = F.mse_loss(out_mag, tgt_mag)
            loss_pha = F.mse_loss(out_pha ,tgt_pha)
            loss_cmplx = F.mse_loss(out_cmplx,img_gt.to(args.device))
        
            optimizer_mag.zero_grad()
            optimizer_pha.zero_grad()
        
            loss_cmplx.backward()

            optimizer_mag.step()
            optimizer_pha.step()
            
            avg_loss_mag = 0.99 * avg_loss_mag + 0.01 * loss_mag.item() if iter > 0 else loss_mag.item()
            avg_loss_pha = 0.99 * avg_loss_pha + 0.01 * loss_pha.item() if iter > 0 else loss_pha.item()
            avg_loss_cmplx = 0.99 * avg_loss_cmplx + 0.01 * loss_cmplx.item() if iter > 0 else loss_cmplx.item()
        
            writer.add_scalar('TrainLoss_mag', loss_mag.item(), global_step + iter)
            writer.add_scalar('TrainLoss_pha', loss_pha.item(), global_step + iter)
            writer.add_scalar('TrainLoss_cmplx', loss_cmplx.item(), global_step + iter)
            
            if iter % args.report_interval == 0:
                logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss_mag = {loss_mag.item():.4g} Avg Loss Mag = {avg_loss_mag:.4g} '
                f'Loss_pha = {loss_pha.item():.4g} Avg Loss Pha = {avg_loss_pha:.4g} '
                f'Loss_cmplx = {loss_cmplx.item():.4g} Avg Loss cmplx = {avg_loss_cmplx:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            
            start_iter = time.perf_counter()
            
 
    
        
        elif (epoch >= 2 and epoch < 3):
            
            model_mag.eval()
            model_pha.eval()
            model_vs.train()
            
            loss_mag = F.mse_loss(out_mag, tgt_mag)
            loss_pha = F.mse_loss(out_pha ,tgt_pha)
            
            out_cmplx = model_vs(out_cmplx,ksp_us,sens,mask)
            
            loss_cmplx = F.mse_loss(out_cmplx,img_gt.cuda())
            
            optimizer_vs.zero_grad()
        
            loss_cmplx.backward()
        
            optimizer_vs.step()
            
            avg_loss_mag = 0.99 * avg_loss_mag + 0.01 * loss_mag.item() if iter > 0 else loss_mag.item()
            avg_loss_pha = 0.99 * avg_loss_pha + 0.01 * loss_pha.item() if iter > 0 else loss_pha.item()
            avg_loss_cmplx = 0.99 * avg_loss_cmplx + 0.01 * loss_cmplx.item() if iter > 0 else loss_cmplx.item()
                       
            writer.add_scalar('TrainLoss_mag', loss_mag.item(), global_step + iter)
            writer.add_scalar('TrainLoss_pha', loss_pha.item(), global_step + iter)
            writer.add_scalar('TrainLoss_cmplx', loss_cmplx.item(), global_step + iter)

            if iter % args.report_interval == 0:
                logging.info(
                    f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                    f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                    f'Loss_mag = {loss_mag.item():.4g} Avg Loss Mag = {avg_loss_mag:.4g} '
                    f'Loss_pha = {loss_pha.item():.4g} Avg Loss Pha = {avg_loss_pha:.4g} '
                    f'Loss_cmplx = {loss_cmplx.item():.4g} Avg Loss cmplx = {avg_loss_cmplx:.4g} '
                    f'Time = {time.perf_counter() - start_iter:.4f}s',
                )
            start_iter = time.perf_counter()
        
        else:
            
            model_mag.train()
            model_pha.train()
            model_vs.train()
            
            out_cmplx = model_vs(out_cmplx,ksp_us,sens,mask)
                   
            loss_mag = F.mse_loss(out_mag, tgt_mag)
            loss_pha = F.mse_loss(out_pha ,tgt_pha)
            loss_cmplx = F.mse_loss(out_cmplx,img_gt.cuda())

            optimizer_mag.zero_grad()
            optimizer_pha.zero_grad()            
            optimizer_vs.zero_grad()
        
            loss_cmplx.backward()
        
            optimizer_mag.step()
            optimizer_pha.step()
            optimizer_vs.step()
            
            avg_loss_mag = 0.99 * avg_loss_mag + 0.01 * loss_mag.item() if iter > 0 else loss_mag.item()
            avg_loss_pha = 0.99 * avg_loss_pha + 0.01 * loss_pha.item() if iter > 0 else loss_pha.item()
            avg_loss_cmplx = 0.99 * avg_loss_cmplx + 0.01 * loss_cmplx.item() if iter > 0 else loss_cmplx.item()
            
            writer.add_scalar('TrainLoss_mag', loss_mag.item(), global_step + iter)
            writer.add_scalar('TrainLoss_pha', loss_pha.item(), global_step + iter)
            writer.add_scalar('TrainLoss_cmplx', loss_cmplx.item(), global_step + iter)

            if iter % args.report_interval == 0:
                logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss_mag = {loss_mag.item():.4g} Avg Loss Mag = {avg_loss_mag:.4g} '
                f'Loss_pha = {loss_pha.item():.4g} Avg Loss Pha = {avg_loss_pha:.4g} '
                f'Loss_cmplx = {loss_cmplx.item():.4g} Avg Loss cmplx = {avg_loss_cmplx:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
            
            
            
    return avg_loss_mag , avg_loss_pha , avg_loss_cmplx,  time.perf_counter() - start_epoch


def evaluate(args, epoch, model_mag , model_pha , model_vs ,  data_loader, writer):
    model_mag.eval()
    model_pha.eval()
    model_vs.eval()
    
    losses_mag = []
    losses_pha = []
    losses_cmplx = []
    start = time.perf_counter()
    with torch.no_grad():

        for iter, data in enumerate(tqdm(data_loader)):
            
            mag_us,mag_gt,pha_us,pha_gt,ksp_us ,img_us,img_gt,img_us_np,img_gt_np,sens,mask,_,_ = data
            
            inp_mag = mag_us.unsqueeze(1).to(args.device)
            tgt_mag = mag_gt.unsqueeze(1).to(args.device)
            inp_pha = pha_us.unsqueeze(1).to(args.device)
            tgt_pha = pha_gt.unsqueeze(1).to(args.device)
            ksp_us = ksp_us.to(args.device)
            sens = sens.to(args.device)
            mask = mask.to(args.device)
            img_gt = img_gt.to(args.device)
                
            out_mag = model_mag(inp_mag)
            out_pha = model_pha(inp_pha)
            
            out_mag_unpad = T.unpad(out_mag , ksp_us)
            out_pha_unpad = T.unpad(out_pha , ksp_us)
            out_cmplx = T.dc(out_mag,out_pha,ksp_us,sens,mask)     
                  
            
            if (epoch >= 2):
                out_cmplx = model_vs(out_cmplx,ksp_us,sens,mask)

            loss_mag = F.mse_loss(out_mag , tgt_mag)
            loss_pha = F.mse_loss(out_pha , tgt_pha)
            loss_cmplx = F.mse_loss(out_cmplx,img_gt.to(args.device))
            
            losses_mag.append(loss_mag.item())
            losses_pha.append(loss_pha.item())
            losses_cmplx.append(loss_cmplx.item())
     
        writer.add_scalar('Dev_Loss_Mag', loss_mag, epoch)
        writer.add_scalar('Dev_Loss_Pha', loss_pha, epoch)
        writer.add_scalar('Dev_Loss_cmplx',loss_cmplx, epoch)
        
    return np.mean(losses_mag), np.mean(losses_pha), np.mean(losses_cmplx) , time.perf_counter() - start


def visualize(args, epoch, model_mag, model_pha, model_vs,  data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model_mag.eval()
    model_mag.eval()
    model_vs.eval()
    
    with torch.no_grad():

        for iter, data in enumerate(data_loader):
            
            mag_us,mag_gt,pha_us,pha_gt,ksp_us ,img_us,img_gt,img_us_np,img_gt_np,sens,mask,_,_ = data
            
            inp_mag = mag_us.unsqueeze(1).to(args.device)
            tgt_mag = mag_gt.unsqueeze(1).to(args.device)
            inp_pha = pha_us.unsqueeze(1).to(args.device)
            tgt_pha = pha_gt.unsqueeze(1).to(args.device)
            ksp_us = ksp_us.to(args.device)
            sens = sens.to(args.device)
            mask = mask.to(args.device)
            img_gt = img_gt.to(args.device)
                
            out_mag = model_mag(inp_mag)
            out_pha = model_pha(inp_pha)
            
            out_mag_unpad = T.unpad(out_mag , ksp_us)
            out_pha_unpad = T.unpad(out_pha , ksp_us)     
            out_cmplx = T.dc(out_mag,out_pha,ksp_us,sens,mask)     
                  
            
            if (epoch >= 2):
                out_cmplx = model_vs(out_cmplx,ksp_us,sens,mask)
            
            save_image(inp_mag, 'Img_mag')
            save_image(tgt_mag, 'Tgt_mag')
            save_image(inp_pha, 'Img_pha')
            save_image(tgt_pha, 'Tgt_pha')
            

            img_gt_cmplx_abs = (torch.sqrt(img_gt[:,:,:,0]**2 + img_gt[:,:,:,1]**2)).unsqueeze(1).to(args.device)
            out_cmplx_abs = (torch.sqrt(out_cmplx[:,:,:,0]**2 + out_cmplx[:,:,:,1]**2)).unsqueeze(1).to(args.device)
            error_cmplx = torch.abs(out_cmplx.cuda() - img_gt.cuda())
            error_cmplx_abs = (torch.sqrt(error_cmplx[:,:,:,0]**2 + error_cmplx[:,:,:,1]**2)).unsqueeze(1).to(args.device)
            
           
            out_cmplx_abs  = T.pad(out_cmplx_abs[0,0,:,:],[256,256]).unsqueeze(0).unsqueeze(1).to(args.device)  
            error_cmplx_abs  = T.pad(error_cmplx_abs[0,0,:,:],[256,256]).unsqueeze(0).unsqueeze(1).to(args.device) 
            img_gt_cmplx_abs  = T.pad(img_gt_cmplx_abs[0,0,:,:],[256,256]).unsqueeze(0).unsqueeze(1).to(args.device) 

            save_image(error_cmplx_abs,'Error')
            save_image(out_cmplx_abs, 'Recons')
            save_image(img_gt_cmplx_abs,'Target')
            
            

            break


def save_model(args, exp_dir, epoch, model_mag , model_pha,model_vs, optimizer_mag, optimizer_pha ,optimizer_vs, best_dev_loss_mag,best_dev_loss_pha,best_dev_loss_cmplx, is_new_best_mag, is_new_best_pha,is_new_best_cmplx):

    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model_mag': model_mag.state_dict(),
            'model_pha': model_pha.state_dict(),
            'model_vs':model_vs.state_dict(),
            'optimizer_mag': optimizer_mag.state_dict(),
            'optimizer_pha': optimizer_pha.state_dict(),
            'optimizer_vs': optimizer_vs.state_dict(), 
            'best_dev_loss_mag': best_dev_loss_mag,
            'best_dev_loss_pha': best_dev_loss_pha,
            'best_dev_loss_cmplx': best_dev_loss_cmplx,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'unet_vs_model.pt'
    )
    if is_new_best_cmplx:   
        shutil.copyfile(exp_dir / 'unet_vs_model.pt', exp_dir / 'best_unet_vs_model.pt')

def build_unet(args):
    print("device",args.device)
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


def load_model(checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model_mag , model_pha , model_vs = build_model(args)
    if args.data_parallel:
        model_mag = torch.nn.DataParallel(model_mag)
        model_pha = torch.nn.DataParallel(model_pha)
        model_vs = torch.nn.DataParallel(model_vs)
        
    model_mag.load_state_dict(checkpoint['model_mag'])
    model_pha.load_state_dict(checkpoint['model_pha'])
    model_vs.load_state_dict(checkpoint['model_vs'])
    

    optimizer_mag = build_optim(args, model_mag.parameters())
    optimizer_pha = build_optim(args, model_pha.parameters())
    optimizer_vs = build_optim(args, model_vs.parameters())

    
    optimizer_mag.load_state_dict(checkpoint['optimizer_mag'])
    optimizer_pha.load_state_dict(checkpoint['optimizer_pha'])
    optimizer_vs.load_state_dict(checkpoint['optimizer_vs'])
    
    
    
    return checkpoint, model_mag , model_pha , model_vs ,  optimizer_mag , optimizer_pha , optimizer_vs


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'unet_vs_summary')

    if args.resume == 'True':

        checkpoint, model_mag, model_pha ,model_vs, optimizer_mag, optimizer_pha, optimizer_vs = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss_mag = checkpoint['best_dev_loss_mag']
        best_dev_loss_pha = checkpoint['best_dev_loss_pha']
        best_dev_loss_cmplx = checkpoint['best_dev_loss_cmplx']
        start_epoch = checkpoint['epoch']

        del checkpoint
    else:
        model_mag , model_pha , model_vs = build_model(args)

        if args.data_parallel:
            model_mag = torch.nn.DataParallel(model_mag)
            model_pha = torch.nn.DataParallel(model_pha)
            model_vs = torch.nn.DataParallel(model_vs)
            
        optimizer_mag = build_optim(args, model_mag.parameters())
        optimizer_pha = build_optim(args, model_pha.parameters())
        optimizer_vs = build_optim(args, model_vs.parameters())
        
        
        best_dev_loss_mag = 1e9
        best_dev_loss_pha = 1e9
        best_dev_loss_cmplx = 1e9
        start_epoch = 0
        
    logging.info(args)
    logging.info(model_mag)
    logging.info(model_pha)
    logging.info(model_vs)

    train_loader, dev_loader , display_loader = create_data_loaders(args)     
    scheduler_mag = torch.optim.lr_scheduler.StepLR(optimizer_mag, args.lr_step_size, args.lr_gamma)
    scheduler_pha = torch.optim.lr_scheduler.StepLR(optimizer_pha, args.lr_step_size, args.lr_gamma)
    scheduler_vs = torch.optim.lr_scheduler.StepLR(optimizer_vs, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        
        scheduler_mag.step(epoch)
        scheduler_pha.step(epoch)
        scheduler_vs.step(epoch)

        train_loss_mag , train_loss_pha, train_loss_cmplx, train_time = train_epoch(args, epoch, model_mag, model_pha, model_vs , train_loader, optimizer_mag , optimizer_pha,optimizer_vs, writer)
        dev_loss_mag, dev_loss_pha , dev_loss_cmplx , dev_time = evaluate(args, epoch, model_mag , model_pha , model_vs ,  dev_loader, writer)
        visualize(args, epoch, model_mag, model_pha, model_vs ,  display_loader, writer)

        is_new_best_mag = dev_loss_mag < best_dev_loss_mag
        is_new_best_pha = dev_loss_pha < best_dev_loss_pha
        is_new_best_cmplx = dev_loss_cmplx < best_dev_loss_cmplx
        
        
        best_dev_loss_mag = min(best_dev_loss_mag, dev_loss_mag)
        best_dev_loss_pha = min(best_dev_loss_pha, dev_loss_pha)
        best_dev_loss_cmplx = min(best_dev_loss_cmplx, dev_loss_cmplx)
                
        save_model(args, args.exp_dir, epoch, model_mag , model_pha, model_vs, optimizer_mag, optimizer_pha, optimizer_vs, best_dev_loss_mag, best_dev_loss_pha, best_dev_loss_cmplx , is_new_best_mag, is_new_best_pha,is_new_best_cmplx)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss_mag = {train_loss_mag:.4g} TrainLoss_pha = {train_loss_pha:.4g} TrainLoss_cmplx = {train_loss_cmplx:.4g}  '
            f'DevLoss_mag = {dev_loss_mag:.4g}  DevLoss_pha = {dev_loss_pha:.4g}  DevLoss_cmplx = {dev_loss_cmplx:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():
    parser = Args()
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
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
    parser.add_argument('--data-path', type=pathlib.Path,help='Path to the dataset')
    parser.add_argument('--checkpoint', type=str,help='Path to an existing checkpoint. Used along with "--resume"')


    parser.add_argument('--residual', type=str, default='False')
    parser.add_argument('--acceleration', type=int,help='Ratio of k-space columns to be sampled. 5x or 10x masks provided')
    
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
        
    main(args)
