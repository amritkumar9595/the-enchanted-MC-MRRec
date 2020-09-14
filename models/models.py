import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import itertools
from data.transforms import ifft2,fft2,fftshift
import data.transforms as T
import math

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,residual='False'):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.residual=residual
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input

        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)
        if self.residual=='True':
            return self.conv2(output)+input
        else:
            return self.conv2(output)
        
            

class DataConsistencyLayer(nn.Module):

    def __init__(self,mask_path,acc_factor,device):
        
        super(DataConsistencyLayer,self).__init__()

        print (mask_path)
        mask_path = os.path.join(mask_path,'mask_{}.npy'.format(acc_factor))
        self.mask = torch.from_numpy(np.load(mask_path)).unsqueeze(2).unsqueeze(0).to(device)

    def forward(self,us_kspace,predicted_img):

        # us_kspace     = us_kspace[:,0,:,:]
        predicted_img = predicted_img[:,0,:,:]
        
        kspace_predicted_img = torch.rfft(predicted_img,2,True,False).double()
        # print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,self.mask.shape)
        
        updated_kspace1  = self.mask * us_kspace 
        updated_kspace2  = (1 - self.mask) * kspace_predicted_img

        

        updated_kspace   = updated_kspace1[:,0,:,:,:] + updated_kspace2
        
        
        updated_img    = torch.ifft(updated_kspace,2,True) 
        
        update_img_abs = torch.sqrt(updated_img[:,:,:,0]**2 + updated_img[:,:,:,1]**2)
        
        update_img_abs = update_img_abs.unsqueeze(1)
        
        return update_img_abs.float()

    

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        #res1
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()
        #res1
        #concat1

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu6 = nn.PReLU()

        #res2
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()
        #res2
        #concat2

        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu10 = nn.PReLU()

        #res3
        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()
        #res3

        self.conv13 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.up14 = nn.PixelShuffle(2)

        #concat2
        self.conv15 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        #res4
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu17 = nn.PReLU()
        #res4

        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.up19 = nn.PixelShuffle(2)

        #concat1
        self.conv20 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        #res5
        self.conv21 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu22 = nn.PReLU()
        self.conv23 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu24 = nn.PReLU()
        #res5

        self.conv25 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)




    def forward(self, x):
        res1 = x
        out = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        out = torch.add(res1, out)
        cat1 = out

        out = self.relu6(self.conv5(out))
        res2 = out
        out = self.relu8(self.conv7(out))
        out = torch.add(res2, out)
        cat2 = out

        out = self.relu10(self.conv9(out))
        res3 = out

        out = self.relu12(self.conv11(out))
        out = torch.add(res3, out)

        out = self.up14(self.conv13(out))

        out = torch.cat([out, cat2], 1)
        out = self.conv15(out)
        res4 = out
        out = self.relu17(self.conv16(out))
        out = torch.add(res4, out)

        out = self.up19(self.conv18(out))

        out = torch.cat([out, cat1], 1)
        out = self.conv20(out)
        res5 = out
        out = self.relu24(self.conv23(self.relu22(self.conv21(out))))
        out = torch.add(res5, out)

        out = self.conv25(out)
        out = torch.add(out, res1)

        return out

class Recon_Block(nn.Module):
    def __init__(self):
        super(Recon_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6= nn.PReLU()
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()

        self.conv13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu14 = nn.PReLU()
        self.conv15 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu16 = nn.PReLU()

        self.conv17 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        res1 = x
        output = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        output = torch.add(output, res1)

        res2 = output
        output = self.relu8(self.conv7(self.relu6(self.conv5(output))))
        output = torch.add(output, res2)

        res3 = output
        output = self.relu12(self.conv11(self.relu10(self.conv9(output))))
        output = torch.add(output, res3)

        res4 = output
        output = self.relu16(self.conv15(self.relu14(self.conv13(output))))
        output = torch.add(output, res4)

        output = self.conv17(output)
        output = torch.add(output, res1)

        return output
    
## Pro version of DUNet   
    
class _NetG(nn.Module):
    def __init__(self):
        super(_NetG, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()   ##it was PRelu
        self.conv_down = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.PReLU()

        self.recursive_A = _Residual_Block()
        self.recursive_B = _Residual_Block()
        self.recursive_C = _Residual_Block()
        # self.recursive_D = _Residual_Block()
        # self.recursive_E = _Residual_Block()
        # self.recursive_F = _Residual_Block()

        self.recon = Recon_Block()
        #concat

        # self.conv_mid = nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_mid = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu3 = nn.PReLU()
        self.conv_mid2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.subpixel = nn.PixelShuffle(2)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)




    def forward(self, x):

        residual = x
        out = self.relu1(self.conv_input(x))
        # print("out1",out.max())
        out = self.relu2(self.conv_down(out))
        # print("out2",out.max())

        out1 = self.recursive_A(out)
        out2 = self.recursive_B(out1)
        out3 = self.recursive_C(out2)
        # out4 = self.recursive_D(out3)
        # out5 = self.recursive_E(out4)
        # out6 = self.recursive_F(out5)

        recon1 = self.recon(out1)
        recon2 = self.recon(out2)
        recon3 = self.recon(out3)
        # recon4 = self.recon(out4)
        # recon5 = self.recon(out5)
        # recon6 = self.recon(out6)
        
        out = torch.cat([recon1, recon2, recon3], 1) 

        # out = torch.cat([recon1, recon2, recon3, recon4, recon5, recon6], 1)
        # out = torch.cat([recon1], 1)

        out = self.relu3(self.conv_mid(out))
        residual2 = out
        out = self.relu4(self.conv_mid2(out))
        out = torch.add(out, residual2)

        out= self.subpixel(out)
        out = self.conv_output(out)
        out = torch.add(out, residual)

        return out


## DUNet lite version 

class _NetG_lite(nn.Module):
    def __init__(self):
        super(_NetG_lite, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()   ##it was PRelu
        self.conv_down = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.PReLU()

        self.recursive_A = _Residual_Block()
#         self.recursive_B = _Residual_Block()
#         self.recursive_C = _Residual_Block()
#         self.recursive_D = _Residual_Block()
#         self.recursive_E = _Residual_Block()
#         self.recursive_F = _Residual_Block()

        self.recon = Recon_Block()
        #concat

        # self.conv_mid = nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_mid = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu3 = nn.PReLU()
        self.conv_mid2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.subpixel = nn.PixelShuffle(2)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)




    def forward(self, x):
        # print("x",x.max())
        residual = x
        out = self.relu1(self.conv_input(x))
        # print("out1",out.max())
        out = self.relu2(self.conv_down(out))
        # print("out2",out.max())

        out1 = self.recursive_A(out)
#         out2 = self.recursive_B(out1)
#         out3 = self.recursive_C(out2)
#         out4 = self.recursive_D(out3)
#         out5 = self.recursive_E(out4)
#         out6 = self.recursive_F(out5)

        recon1 = self.recon(out1)
#         recon2 = self.recon(out2)
#         recon3 = self.recon(out3)
#         recon4 = self.recon(out4)
#         recon5 = self.recon(out5)
#         recon6 = self.recon(out6)

#         out = torch.cat([recon1, recon2, recon3, recon4, recon5, recon6], 1)
        out = torch.cat([recon1], 1)

        out = self.relu3(self.conv_mid(out))
        residual2 = out
        out = self.relu4(self.conv_mid2(out))
        out = torch.add(out, residual2)

        out= self.subpixel(out)
        out = self.conv_output(out)
        out = torch.add(out, residual)

        return out
    
## VS-Net architecture
    
class dataConsistencyTerm(nn.Module):
    
    def __init__(self, noise_lvl=None):
        super(dataConsistencyTerm, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            self.noise_lvl = torch.nn.Parameter(torch.Tensor([noise_lvl]))

    def perform(self,out_img_cmplx,ksp,sens,mask ):
        
        x = T.complex_multiply(out_img_cmplx[...,0].unsqueeze(1), out_img_cmplx[...,1].unsqueeze(1), sens[...,0], sens[...,1])
    
        k = (torch.fft(x, 2, normalized=True)).squeeze(1)
        k_shift = T.ifftshift(k, dim=(-3,-2))

        
        sr = 0.85
        Nz = k_shift.shape[-2] 
        Nz_sampled = int(np.ceil(Nz*sr))
        k_shift[:,:,:,Nz_sampled:,:] = 0
        
        v = self.noise_lvl
        if v is not None: # noisy case
            # out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
            out = (1 - mask) * k_shift + mask * (v * k_shift + (1 - v) * ksp) 
        
        else:
            
            out = (1 - mask) * k_shift + mask * ksp
        
        
        x = torch.ifft(out, 2, normalized=True)
    
        Sx = T.complex_multiply(x[...,0], x[...,1], sens[...,0],-sens[...,1]).sum(dim=1)
        
        Ss = T.complex_multiply(sens[...,0], sens[...,1], sens[...,0],-sens[...,1]).sum(dim=1)
     
        return Sx, Ss
    

    
class weightedAverageTerm(nn.Module):

    def __init__(self, para=None):
        super(weightedAverageTerm, self).__init__()
        self.para = para
        if para is not None:
            self.para = torch.nn.Parameter(torch.Tensor([para]))

    def perform(self, cnn, Sx, SS):
        
        x = self.para*cnn + (1 - self.para)*Sx
        return x



class cnn_layer(nn.Module):
    
    def __init__(self):
        super(cnn_layer, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(2,  64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2,  3, padding=1, bias=True)
        )     
        
    def forward(self, x):
        
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    
class network(nn.Module):
    
    def __init__(self, alfa=1, beta=1, cascades=5):
        super(network, self).__init__()
        
        self.cascades = cascades 
        conv_blocks = []
        dc_blocks = []
        wa_blocks = []
        
        for i in range(cascades):
            conv_blocks.append(cnn_layer()) 
            dc_blocks.append(dataConsistencyTerm(alfa)) 
            wa_blocks.append(weightedAverageTerm(beta)) 
        
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dc_blocks = nn.ModuleList(dc_blocks)
        self.wa_blocks = nn.ModuleList(wa_blocks)
        
        print(self.conv_blocks)
        print(self.dc_blocks)
        print(self.wa_blocks)
 
    def forward(self, x, k, m, c):
                
        for i in range(self.cascades):
            x_cnn = self.conv_blocks[i](x)
            Sx, SS = self.dc_blocks[i].perform(x, k, m, c)
            x = self.wa_blocks[i].perform(x + x_cnn, Sx, SS)
        return x    


class SSIM(nn.Module):
    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer('w', torch.ones(
            1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (2 * ux * uy + C1, 2 * vxy + C2,
                          ux ** 2 + uy ** 2 + C1, vx + vy + C2)
        D = B1 * B2
        S = (A1 * A2) / D
        return 1 - S.mean()