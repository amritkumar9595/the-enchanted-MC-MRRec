'''
to generate slices from the volume kspace
each slice contains: complex kspace [218, 170, 12] ; key ='kspace' 
                   : complex sensitivity_maps [(12, 218, 170)] ; key = 'sensitivity'
'''

import torch
import pathlib
import h5py
import numpy as np
import os
from tqdm import tqdm

#root = '/media/student1/RemovableVolume/calgary_new/Train/'   ## to generate training slices
root = '/media/student1/RemovableVolume/calgary_new/Val/'

files = list(pathlib.Path(root).iterdir())

examples = []
for fname in sorted(files):

        num_slices = 256 #kspace.shape[0]
        examples += [(fname, slice) for slice in range(50,num_slices-50)]

print("total number of slices = ",len(examples))

for i in tqdm(range(len(examples))):
    #     print("i=",i)
    fname, slice = examples[i]
    with h5py.File(fname, 'r') as data:
            # print("opening ksp")
            kspace = data['kspace'][slice]
            sr = 0.85
            Nz = kspace.shape[1]
            Nz_sampled = int(np.ceil(Nz*sr))

            kspace[:,Nz_sampled:,:] = 0
            ksp_cmplx = kspace[:,:,::2] + 1j*kspace[:,:,1::2]
            
            parts = list(fname.parts)
            parts[4]='calgary_new/sens'
            path=pathlib.Path(*parts)  
            with open(path,'rb') as f:

                sensitivity = np.load(f)[slice]
        
            
            parts = list(fname.parts)
            parts[6]

            parts[-1]
            n = parts[-1][:-2] + str(slice) +'.h5'
            parts = list(fname.parts)
            parts[4] = 'calgary'
            n = parts[-1][:-2] + str(slice) + '.h5'
            parts[-1] = n
            
            path=pathlib.Path(*parts)  
            with h5py.File(str(path), 'w') as f:
                f.create_dataset("kspace",     data=ksp_cmplx)
                f.create_dataset("sensitivity",data=sensitivity)


