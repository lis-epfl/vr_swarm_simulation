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

already_saved = False

class NISStitcher(BaseStitcher):
    def __init__(self):
        super().__init__(device="cpu")  # Initialize the base class

        # self.net.to(device)
        conf ="Neural_Image_Stitching_main/configs/test/NIS_blending.yaml"
        with open(conf, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            print('config loaded.')
            
        self.model, self.H_model = stitch.prepare_validation(self.config)
        self.model.eval()
        self.H_model.eval()

    def NIS_warping(self, ref, tgt, inp_ref, inp_tgt):
        
        b, c, h, w = ref.shape

        inp_ref *=255
        inp_tgt *=255

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

        pred, ref_mask = pred.detach().cpu().numpy()[0], ref_mask.detach().cpu().numpy()[0]

        return pred.transpose(1, 2, 0), ref_mask.transpose(1, 2, 0)

    def NIS_pano(self, images, subset1, subset2):
        global already_saved
        t0=time.time()

        ref, tgt, inp_ref, inp_tgt = resize_pair_images(cv2.flip(images[subset1[0]], 1), cv2.flip(images[subset1[1]], 1))
        h, w = ref.shape[:2]
        ref, tgt, inp_ref, inp_tgt = TransformToTensor(ref, tgt, inp_ref, inp_tgt)
        
        # print("warping right")
        left_warp, left_mask = self.NIS_warping(ref, tgt, inp_ref, inp_tgt)

        # Release GPU memory 
        del ref
        del tgt

        # In the case of GPU Out-of-memory (resize the images to avoid extreme memory usage and warping time)
        ref, tgt, inp_ref, inp_tgt = resize_pair_images(images[subset2[0]], images[subset2[1]])
        ref, tgt, inp_ref, inp_tgt = TransformToTensor(ref, tgt, inp_ref, inp_tgt)
        
        # print("warping right")
        right_warp, right_mask = self.NIS_warping(ref, tgt, inp_ref, inp_tgt)
        
        # Release GPU memory 
        del ref
        del tgt
        pano = ComposeTwoSides(left_warp, right_warp, left_mask, right_mask, size=(h, w))

        if not already_saved:
            already_saved = True
            cv2.imwrite("left_warp.jpg", cv2.cvtColor((left_warp * 255).astype('uint8'), cv2.COLOR_RGB2BGR))
            cv2.imwrite("right_warp.jpg", cv2.cvtColor((right_warp * 255).astype('uint8'), cv2.COLOR_RGB2BGR))
            cv2.imwrite("left_mask.jpg", (left_mask * 255).astype('uint8'))
            cv2.imwrite("right_mask.jpg", (right_mask * 255).astype('uint8'))
            cv2.imwrite("pano.jpg", cv2.cvtColor((pano * 255).astype('uint8'), cv2.COLOR_RGB2BGR))
            print("saving")

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


def resize_pair_images(ref, tgt):
    h, w, _ = ref.shape

    # In the case of GPU Out-of-memory (resize the images to avoid extreme memory usage and warping time)
    if h==w:
        if h>300:
            ref = cv2.resize(ref, dsize=(300, 300), interpolation=cv2.INTER_AREA)
            tgt = cv2.resize(tgt, dsize=(300, 300), interpolation=cv2.INTER_AREA)
    else:
        if h>300 or w>300:
            if w>h:
                new_h = int(300/w*h)
                ref = cv2.resize(ref, dsize=(new_h, 300), interpolation=cv2.INTER_AREA)
                tgt = cv2.resize(tgt, dsize=(new_h, 300), interpolation=cv2.INTER_AREA)
            else:
                new_w = int(300/h*w)
                ref = cv2.resize(ref, dsize=(300, new_w), interpolation=cv2.INTER_AREA)
                tgt = cv2.resize(tgt, dsize=(300, new_w), interpolation=cv2.INTER_AREA)

    if h != 128 or w != 128:
        inp_ref = cv2.resize(ref, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
        inp_tgt = cv2.resize(tgt, dsize=(128,128), interpolation=cv2.INTER_CUBIC)

    return ref, tgt, inp_ref, inp_tgt

def TransformToTensor(ref, tgt, inp_ref, inp_tgt):
    
    ref = transforms.ToTensor()(
        Image.fromarray(ref).convert('RGB')
    ).unsqueeze(0).cuda()

    tgt = transforms.ToTensor()(
        Image.fromarray(tgt).convert('RGB')
    ).unsqueeze(0).cuda()

    inp_ref = transforms.ToTensor()(
        Image.fromarray(inp_ref).convert('RGB')
    ).unsqueeze(0).cuda()

    inp_tgt = transforms.ToTensor()(
        Image.fromarray(inp_tgt).convert('RGB')
    ).unsqueeze(0).cuda()

    return ref, tgt, inp_ref, inp_tgt

def find_image_shift(part_mask):
    """
    Determines the shift of the first image in a panorama using its mask.
    
    Parameters:
        part_mask (numpy.ndarray): A part of the mask of the right/left image in the panorama.
        
    Returns:
        tuple: (y_shift_up, y_shift_down) where  y_shift_up is the vertical shift from above (row index)
               and y_shift_down is the vertical shift from bottom(row index).
    """
    # Ensure the mask is binary (if it's not already)
    binary_mask = part_mask > 0

    # Find the row indices with any non-zero elements
    row_indices = np.any(binary_mask, axis=1)
    y_shift_up = np.argmax(row_indices)  # First row with a non-zero value
    # y_shift_down = np.argmin(row_indices[y_shift_up:])  # First row with a non-zero value

    # # Find the column indices with any non-zero elements
    # col_indices = np.any(binary_mask, axis=0)
    # x_shift = np.argmax(col_indices)  # First column with a non-zero value

    return y_shift_up#, y_shift_down

def ComposeTwoSides(left_warp, right_warp, left_mask, right_mask, size=(300, 300)):
    h, w = size
    rightSize = right_warp.shape
    leftSize = left_warp.shape

    # _, shiftup1 = find_image_shift(left_mask[:, :w//3])
    # _, shiftup2 = find_image_shift(right_mask[:, :w//3])
    # shiftup1, shiftdown1 = find_image_shift(left_mask[:, :w//3])
    # shiftup2, shiftdown2 = find_image_shift(right_mask[:, :w//3])
    shiftup1 = find_image_shift(left_mask[:, :w//3])
    shiftup2 = find_image_shift(right_mask[:, :w//3])
    shiftdown1 = leftSize[0]-h-shiftup1
    shiftdown2 = rightSize[0]-h-shiftup2

    left_warp = cv2.flip(left_warp, 1)

    # Compute difference size between ref image and warped ones
    diff2x, diff2y = rightSize[1]-w, rightSize[0]-h
    diff1x, diff1y = leftSize[1]-w, leftSize[0]-h

    
    print(f"rightSIze: {rightSize}")
    print(f"leftSIze: {leftSize}")
    print(f"diff1x: {diff1x}, diff1y: {diff1y}, shiftup1: {shiftup1}, shiftdown1: {shiftdown1}")
    print(f"diff2x: {diff2x}, diff2y: {diff2y}, shiftup2: {shiftup2}, shiftdown2: {shiftdown2}")
    print(f"shiftup1+(leftSize[0]-shiftdown1): {shiftup1+(leftSize[0]-shiftdown1)}")
    print(f"shiftup2+(rightSize[0]-shiftdown2): {shiftup2+(rightSize[0]-shiftdown2)}")
    
    pano = np.zeros((diff1y+diff2y+h, diff1x+diff2x+w, 3))
    # pano = np.zeros((h, diff1x+diff2x+w, 3))

    if diff2y == shiftup2+shiftdown2 and diff1y == shiftup1+shiftdown1:
        diffshiftup = shiftup2-shiftup1

        if diffshiftup >= 0:
            pano[diffshiftup:leftSize[0]+diffshiftup, :diff1x+w//2] = left_warp[:, :diff1x+w//2]
            pano[:rightSize[0], diff1x+w//2:] = right_warp[:, w//2:]
        else:
            pano[:leftSize[0], :diff1x+w//2] = left_warp[:, :diff1x+w//2]
            # Problem dimension here
            pano[-diffshiftup:rightSize[0]-diffshiftup, diff1x+w//2:] = right_warp[:, w//2:]
            
    else:
        print("Control if shifts are equal:")
        # print(diff2y==shiftup2+shiftdown2, diff1y==shiftup1+shiftdown1)
        
        # print("Alignement problem")
        # print(f"diff1x: {diff1x}, diff1y: {diff1y}, shiftup1: {shiftup1}, shiftdown1: {shiftdown1}")
        # print(f"diff2x: {diff2x}, diff2y: {diff2y}, shiftup2: {shiftup2}, shiftdown2: {shiftdown2}")
        
        # If there are errors in the calculation due to bad image order or bad logical calculations
        pano = right_warp

    return pano#.astype(np.uint8)