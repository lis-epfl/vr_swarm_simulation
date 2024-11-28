# import custom_stitching
import numpy as np
import cv2
import torch
import time
from BaseStitcher import *


import glob
import os
from numba import jit

import sys
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from PIL import Image
import os

sys.path.append(os.path.abspath("Neural_Image_Stitching_main"))
import Neural_Image_Stitching_main.srwarp
import Neural_Image_Stitching_main.utils
from Neural_Image_Stitching_main.models.ihn import *
from Neural_Image_Stitching_main.models import *
from Neural_Image_Stitching_main import stitch
import Neural_Image_Stitching_main.pretrained
import yaml
from types import SimpleNamespace
import torch.nn.functional as F
from torchvision import transforms

# from transformers import SuperPointForKeypointDetection
# # from torch.quantization import quantize_dynamic
# from numba import jit

class NISStitcher(BaseStitcher):
    def __init__(self):
        super().__init__(device="cpu")  # Initialize the base class

        # self.net.to(device)
        conf ="Neural_Image_Stitching_main/configs/test/NIS_blending.yaml"
        with open(conf, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            print('config loaded.')
            
        self.model, self.H_model = stitch.prepare_validation(self.config)

    def NIS_warping(self, reference, target):
        self.model.eval()
        self.H_model.eval()

        b, c, h, w = reference.shape

        # In the case of GPU Out-of-memory
        # TODO change this and maybe look at a proportional factor or just squared images
        ref = F.interpolate(reference, size=(480, 640), mode='area')
        tgt = F.interpolate(target, size=(480, 640), mode='area')

        print(ref.shape)

        if ref.shape[-2:] != tgt.shape[-2:]:
            tgt = F.interpolate(tgt, size=(h, w), mode='bilinear')

        if h != 128 or w != 128:
            inp_ref = F.interpolate(ref, size=(128,128), mode='bilinear') * 255
            inp_tgt = F.interpolate(tgt, size=(128,128), mode='bilinear') * 255

        else:
            inp_ref = ref * 255
            inp_tgt = tgt * 255

        tgt_grid, tgt_cell, tgt_mask, \
        ref_grid, ref_cell, ref_mask, \
        stit_grid, stit_mask, sizes = stitch.prepare_ingredient(self.H_model, inp_tgt, inp_ref, tgt, ref)

        ref = (ref - 0.5) * 2
        tgt = (tgt - 0.5) * 2

        ref_mask = ref_mask.reshape(b,1,*sizes)
        tgt_mask = tgt_mask.reshape(b,1,*sizes)

        pred = stitch.batched_predict(
            self.model, ref, ref_grid, ref_cell, ref_mask,
            tgt, tgt_grid, tgt_cell, tgt_mask,
            stit_grid, sizes, self.config['eval_bsize'], seam_cut=True
        )
        pred = pred.permute(0, 2, 1).reshape(b, c, *sizes)
        pred = ((pred + 1)/2).clamp(0,1) * stit_mask.reshape(b, 1, *sizes)

        return pred

    def NIS_pano(self, images, subset1, subset2):
        t0=time.time()
        # ref, tgt = torch.from_numpy(images[subset2[0]]).float().permute(2, 0, 1).unsqueeze(0).cuda(), torch.from_numpy(images[subset2[1]]).float().permute(2, 0, 1).unsqueeze(0).cuda()
        ref = transforms.ToTensor()(
            Image.fromarray(images[subset2[0]]).convert('RGB')
        ).unsqueeze(0).cuda()

        tgt = transforms.ToTensor()(
            Image.fromarray(images[subset2[1]]).convert('RGB')
        ).unsqueeze(0).cuda()
        
        # print("warping right")
        right_warp = self.NIS_warping(ref, tgt)

        pano = right_warp[0].detach().cpu().numpy()
        pano = pano.transpose(1, 2, 0)

        return (pano * 255).astype('uint8')#.astype(np.uint8)
  
    def stitch(self, images, order, Hs, inverted, headAngle, num_pano_img=3, verbose=False):
        """""
        This method uses some of the above methods to stitch a part of the given images based on a criterion that could be the orientation
        of the pilots head and the desired number of images in the panorama.
        Input:
            - images: list of NDArrays.
            - angle : orientation of the pilots head (in degrees [0,360[?)
            - num_pano_img : desired number of images in the panorama
        """""
        # Try taking the homography and order. Until the queues are empty, keep the homograpies and orders in local variable
        # Take the order and the ref to compute the panorama

        t = time.time()

        subset1, subset2, _, _ = self.chooseSubsetsAndTransforms(Hs, num_pano_img, order, headAngle)
        with torch.no_grad():
            pano = self.NIS_pano(images, subset1, subset2)          
        
        if verbose:
            print(f"Warp time: {time.time()-t}")
        return pano
    
