import numpy as np
import torch
import math
from torch.nn import functional as F


def complex_multiply(x, y, u, v):
    """
    Computes (x+iy) * (u+iv) = (x * u - y * v) + (x * v + y * u)i = z1 + iz2
    
    Returns (real z1, imaginary z2)
    """

    z1 = x * u - y * v
    z2 = x * v + y * u
    
    return torch.stack((z1, z2), dim=-1)

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        # print("before to_tensor",data.shape)
        data = np.stack((data.real, data.imag), axis=-1)
    # print("to_tensor",data.shape)
    return torch.from_numpy(data)


def apply_mask(data, mask_func, seed=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """ 
    # print("in trnsforms apply_mask")
    shape = np.array(data.shape)
    shape[:-3] = 1
    # print("mask_func",mask_func)
    mask = mask_func(shape, seed)
    print("mask_transforms",mask.shape)
    return torch.where(mask == 0, torch.Tensor([0]), data), mask


def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    # data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    # data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    # data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    # data = fftshift(data, dim=(-3, -2))
    # print("data",data.shape)
    return data


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data, shape):
    print("centre_crop",data.shape)
    print("shape0",shape)
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


# Helper functions

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)



def combine_all_coils(image, sensitivity):
    
    """return sensitivity combined images from all coils""" 
    combined = complex_multiply(sensitivity[...,0],-sensitivity[...,1],image[...,0],image[...,1]) 
    return combined.sum(dim = 0)

def phase(data):
    
    inp_mag = complex_abs(data)
    phase = torch.atan(data[:,:,1]/data[:,:,0]).masked_fill(inp_mag<=0.04,0)
    return phase

def pad( x,shape):
    
    def floor_ceil(n):
        return math.floor(n), math.ceil(n)
    # print("shape",x.shape)
    h,w = x.shape

    w_pad = floor_ceil((256 - w) / 2)
    h_pad = floor_ceil((256 - h) / 2)
    x = F.pad(x, w_pad + h_pad)
    return x #, (h_pad, w_pad) #h_mult, w_mult)

def unpad(img,x):
    #     x =  x.permute(2,0,1)
    def floor_ceil(n):
        return math.floor(n), math.ceil(n)
    b,c,h,w,_ = x.shape
    h_pad = floor_ceil((256 - h) / 2)
    w_pad = floor_ceil((256 - w) / 2)
    img =  img[..., h_pad[0]:256 - h_pad[1], w_pad[0]:256 - w_pad[1]]
    return img #.permute(1,2,0)
    
def combine_mag_pha(mag , pha):
    
    img_cmplx_real = mag*(torch.cos(pha))
    img_cmplx_imag =  mag*(torch.sin(pha))
    complex_out_img = torch.stack((img_cmplx_real,img_cmplx_imag),dim=-1)
    
    return complex_out_img.squeeze(1)
    

def dc(out_mag,out_pha,ksp,sens,mask ):
    
    out_mag_unpad = unpad(out_mag , ksp)
    out_pha_unpad = unpad(out_pha , ksp)
    
    out_img_cmplx = combine_mag_pha(out_mag_unpad,out_pha_unpad)
    x = complex_multiply(out_img_cmplx[...,0].unsqueeze(1), out_img_cmplx[...,1].unsqueeze(1), sens[...,0].cuda(), sens[...,1].cuda())
    
    k = (torch.fft(x, 2, normalized=True)).squeeze(1)
    k_shift = ifftshift(k, dim=(-3,-2))
    
    sr = 0.85
    Nz = k_shift.shape[-2] 
    Nz_sampled = int(np.ceil(Nz*sr))
    k_shift[:,:,:,Nz_sampled:,:] = 0
    
    out = (1 - mask.cuda()) * k_shift.cuda() + mask.cuda() * ksp.cuda()
    x = torch.ifft(out, 2, normalized=True)
    
    Sx = complex_multiply(x[...,0], x[...,1], sens[...,0].cuda(),-sens[...,1].cuda()).sum(dim=1)
    
    return Sx


def inp_to_dun(img_t):
    """ valid for batch size=1 """
    
    img_mag = torch.sqrt((img_t[0,:,:,0]**2 + img_t[0,:,:,1]**2))
    img_mag = img_mag/img_mag.max()
    img_mag_pad = pad(img_mag,[256,256]).unsqueeze(0).unsqueeze(1)
    
    return img_mag_pad

def out_from_dun(img,x):
    def floor_ceil(n):
        return math.floor(n), math.ceil(n)
    
    b,c,h,w = x.shape
    h_pad = floor_ceil((256 - h) / 2)
    w_pad = floor_ceil((256 - w) / 2)
    img =  img[..., h_pad[0]:256 - h_pad[1], w_pad[0]:256 - w_pad[1]]
    return img #.permute(1,2,0)
    
    


## to find the GT for the final submission


def channel_wise_ifft(zero_filled_kspace):
    """
    Computes the iFFT across channels of multi-channel k-space data. The input is expected to be a complex numpy array.
    """
    return np.fft.ifft2(zero_filled_kspace, axes = (0,1))
    
    

def sum_of_squares(img_channels):
    """
    Combines complex channels with square root sum of squares. The channels are the last dimension (i.e., -1) of the input array.
    """
    return np.sqrt((np.abs(img_channels)**2).sum(axis = -1))
    return sos    

def zero_filled_reconstruction(zero_filled_kspace):
    """
    Zero-filled reconstruction of multi-channel MR images. The input is the zero-filled k-space. The channels
    are the last dimension of the array. The input may be either complex-valued or alternate between real and imaginary channels 
    in the last array dimension.
    """
    if not np.iscomplexobj(zero_filled_kspace):
        zero_filled_kspace = zero_filled_kspace[:,:,:,::2] + 1j*zero_filled_kspace[:,:,:,1::2] #convert real-imag to complex data
    
    img_gt_np = sum_of_squares(channel_wise_ifft(zero_filled_kspace))
    img_gt_np = torch.from_numpy(img_gt_np)
    img_gt_np_pad = pad(img_gt_np,[256,256])
    
    return img_gt_np_pad


def rss(out_img_cmplx,sens):
    
        x = complex_multiply(out_img_cmplx[...,0].unsqueeze(1), out_img_cmplx[...,1].unsqueeze(1), sens[...,0].cuda(), sens[...,1].cuda())
        rs = complex_abs(x)
        rs = root_sum_of_squares(rs,1).squeeze(0)
        rs = rs/rs.max()
        rs = pad(rs,[256,256]).unsqueeze(0).unsqueeze(1)
        
        return rs 
        
    



    
    

    