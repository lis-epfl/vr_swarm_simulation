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
    def __init__(self, onlyIHN = True):
        super().__init__(device="cpu")  # Initialize the base class

        # self.net.to(device)
        conf ="Neural_Image_Stitching_main/configs/test/NIS_blending.yaml"
        with open(conf, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            print('config loaded.')
            
        self.model, self.H_model = stitch.prepare_validation(self.config)
        self.model.eval()
        self.H_model.eval()
        # Not implemented yet (Can maybe use only IHN for stitching and then do simple blending for fast computation)
        self.onlyIHN = onlyIHN

    def NIS_warping(self, ref, tgt, inp_ref, inp_tgt):
        """
        Performs Neural Image Stitching (NIS) warping to blend reference and target images using a deep learning model.

        Parameters:
        - ref (torch.Tensor): The reference image tensor of shape (batch_size, channels, height, width).
        - tgt (torch.Tensor): The target image tensor of shape (batch_size, channels, height, width).
        - inp_ref (torch.Tensor): Preprocessed input reference image tensor (scaled between 0 and 1).
        - inp_tgt (torch.Tensor): Preprocessed input target image tensor (scaled between 0 and 1).

        Returns:
        - pred (np.ndarray): The warped and blended output image of shape (height, width, channels).
        - ref_mask (np.ndarray): The mask for the reference image of shape (height, width, 1).
        """
        
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
        """
        Performs Neural Image Stitching (NIS) to create a panorama from input images using subsets.

        Parameters:
        - images (list): List of input images to be stitched into a panorama.
        - subset1 (np.ndarray): Indices of images on the left side of the panorama.
        - subset2 (np.ndarray): Indices of images on the right side of the panorama.

        Returns:
        - pano (np.ndarray): The stitched panorama image in uint8 format.
        """

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

        return (pano * 255).astype('uint8')#.astype(np.uint8)
  
    def IHN_warping(self, ref, tgt, inp_ref, inp_tgt):
        """
        Performs Image Harmonization Network (IHN) warping to blend reference and target images.

        Parameters:
        - ref (torch.Tensor): The reference image tensor of shape (batch_size, channels, height, width).
        - tgt (torch.Tensor): The target image tensor of shape (batch_size, channels, height, width).
        - inp_ref (torch.Tensor): Preprocessed input reference image tensor (scaled between 0 and 1).
        - inp_tgt (torch.Tensor): Preprocessed input target image tensor (scaled between 0 and 1).

        Returns:
        - canvas_visual (np.ndarray): The stitched and blended canvas image of shape (height, width, channels).
        - ref_mask (np.ndarray): The mask for the reference image of shape (height, width, 1).
        """

        b, c, h, w = ref.shape

        inp_ref *=255
        inp_tgt *=255

        tgt_grid, tgt_cell, tgt_mask, \
        ref_grid, ref_cell, ref_mask, \
        stit_grid, stit_mask, sizes = stitch.prepare_ingredient(self.H_model, inp_tgt, inp_ref, tgt, ref)

        ref_mask = ref_mask.reshape(b,1,*sizes)
        tgt_mask = tgt_mask.reshape(b,1,*sizes)

        output_h, output_w = sizes  # Output size for the stitched canvas

        # Reshape coord to match the output size
        tgt_grid_reshaped = tgt_grid.view(b, output_h, output_w, 2)

        # Flip last dimension of coord to match PyTorch grid_sample convention
        tgt_grid_flipped = tgt_grid_reshaped.flip(-1)

        # Warp the target image using grid_sample
        warped_tgt = torch.nn.functional.grid_sample(
            tgt,  # Normalize input to [0, 1]
            tgt_grid_flipped,  # Use flipped grid
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        # Apply mask
        warped_tgt_masked = warped_tgt * tgt_mask
        
        # Normalize and warp the reference image
        ref_grid_normalized = ref_grid.clone()
        ref_grid_normalized = ref_grid_normalized.view(b, output_h, output_w, 2)
        ref_grid_normalized = ref_grid_normalized.flip(-1)  # Flip last dimension for PyTorch conventions

        # Warp the reference image using grid_sample
        warped_ref = torch.nn.functional.grid_sample(
            ref,  # Normalize input to [0, 1]
            ref_grid_normalized,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        # Apply the reference mask
        warped_ref_masked = warped_ref * ref_mask

        # Combine the reference and target images into the canvas
        canvas = torch.zeros_like(warped_ref)  # Initialize the canvas with zeros
        combined_mask = torch.zeros_like(ref_mask)  # Combined mask for normalization

        # Add reference image to the canvas
        canvas += warped_ref_masked
        combined_mask += ref_mask

        # Add target image to the canvas
        canvas += warped_tgt_masked
        combined_mask += tgt_mask

        # Normalize the canvas by the combined mask
        canvas = canvas / (combined_mask + 1e-5)  # Avoid division by zero

        # Visualize the final stitched canvas
        # canvas_visual = (canvas[0].cpu().clamp(0, 1) * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
        canvas_visual = canvas[0].cpu().clamp(0, 1).numpy().transpose(1, 2, 0)


        return canvas_visual, ref_mask.cpu()[0].numpy().transpose(1, 2, 0)

    def IHN_pano(self, images, subset1, subset2):
        """
        Creates a panorama by stitching images using the Image Harmonization Network (IHN).

        Parameters:
        - images (list): List of input images to be stitched into a panorama.
        - subset1 (np.ndarray): Indices of images on the left side of the panorama.
        - subset2 (np.ndarray): Indices of images on the right side of the panorama.

        Returns:
        - pano (np.ndarray): The final stitched panorama image in uint8 format.
        """

        ref, tgt, inp_ref, inp_tgt = resize_pair_images(cv2.flip(images[subset1[0]], 1), cv2.flip(images[subset1[1]], 1))
        h, w = ref.shape[:2]
        ref, tgt, inp_ref, inp_tgt = TransformToTensor(ref, tgt, inp_ref, inp_tgt)
        
        # print("warping right")
        left_warp, left_mask = self.IHN_warping(ref, tgt, inp_ref, inp_tgt)

        # Release GPU memory 
        del ref
        del tgt

        # In the case of GPU Out-of-memory (resize the images to avoid extreme memory usage and warping time)
        ref, tgt, inp_ref, inp_tgt = resize_pair_images(images[subset2[0]], images[subset2[1]])
        ref, tgt, inp_ref, inp_tgt = TransformToTensor(ref, tgt, inp_ref, inp_tgt)
        
        # print("warping right")
        right_warp, right_mask = self.IHN_warping(ref, tgt, inp_ref, inp_tgt)
        
        # Release GPU memory 
        del ref
        del tgt
        pano = ComposeTwoSides(left_warp, right_warp, left_mask, right_mask, size=(h, w))

        return (pano * 255).astype('uint8')
    
    def stitch(self, images, order, Hs, inverted, headAngle, num_pano_img=3):
        """""
        This method uses some of the above methods to stitch a part of the given images based on a criterion that could be the orientation
        of the pilots head and the desired number of images in the panorama.
        Input:
            - images: list of NDArrays.
            - angle : orientation of the pilots head (in degrees [0,360[?)
            - num_pano_img : desired number of images in the panorama

        Returns:
        - pano (np.ndarray): The final stitched panorama image in uint8 format. If self.onlyIHN is true, only uses IHN without NIS.
        """""
        subset1, subset2, _, _ = self.chooseSubsetsAndTransforms(Hs, num_pano_img, order, headAngle)
        
        if self.onlyIHN:
            with torch.no_grad():
                pano = self.IHN_pano(images, subset1, subset2)  
        else:
            with torch.no_grad():
                pano = self.NIS_pano(images, subset1, subset2)          
        return pano


def resize_pair_images(ref, tgt):
    """
    Resizes a pair of reference and target images to manage memory usage and standardize input dimensions.

    Parameters:
    - ref (np.ndarray): The reference image, a numpy array of shape (height, width, channels).
    - tgt (np.ndarray): The target image, a numpy array of shape (height, width, channels).

    Returns:
    - ref (np.ndarray): The resized reference image.
    - tgt (np.ndarray): The resized target image.
    - inp_ref (np.ndarray): A smaller resized version of the reference image (128x128).
    - inp_tgt (np.ndarray): A smaller resized version of the target image (128x128).
    """
   
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
        inp_ref = cv2.resize(ref, dsize=(128,128), interpolation=cv2.INTER_AREA)
        inp_tgt = cv2.resize(tgt, dsize=(128,128), interpolation=cv2.INTER_AREA)

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
        tuple: y_shift_up where  y_shift_up is the vertical shift from above (row index)
    """
    # Ensure the mask is binary (if it's not already)
    binary_mask = part_mask > 0

    # Find the row indices with any non-zero elements
    row_indices = np.any(binary_mask, axis=1)
    y_shift_up = np.argmax(row_indices)  # First row with a non-zero value

    return y_shift_up

def ComposeTwoSides(left_warp, right_warp, left_mask, right_mask, size=(300, 300), security_factor = 5):
    """
    Composes a panorama by blending warped images from the left and right sides.

    Parameters:
    - left_warp (np.ndarray): The warped image from the left side.
    - right_warp (np.ndarray): The warped image from the right side.
    - left_mask (np.ndarray): The mask for the left warped image.
    - right_mask (np.ndarray): The mask for the right warped image.
    - size (tuple): The size of the reference image (height, width). Default is (300, 300).
    - security_factor (int): A padding factor to avoid dimension mismatches. Default is 5.

    Returns:
    - pano (np.ndarray): The composed panorama image.
    """

    h, w = size
    rightSize = right_warp.shape
    leftSize = left_warp.shape

    shiftup1 = find_image_shift(left_mask[:, :w//3])
    shiftup2 = find_image_shift(right_mask[:, :w//3])
    shiftdown1 = leftSize[0]-h-shiftup1
    shiftdown2 = rightSize[0]-h-shiftup2

    left_warp = cv2.flip(left_warp, 1)

    # Compute difference size between ref image and warped ones
    diff2x, diff2y = rightSize[1]-w, rightSize[0]-h
    diff1x, diff1y = leftSize[1]-w, leftSize[0]-h

    top_shift = max(shiftup1, shiftup2)
    bottom_shift = max((leftSize[0] - shiftup1), (rightSize[0] - shiftup2))
    pano = np.zeros((top_shift + bottom_shift+security_factor, diff1x+diff2x+w, 3))

    if diff2y == shiftup2+shiftdown2 and diff1y == shiftup1+shiftdown1:
        diffshiftup = shiftup2-shiftup1
        try:
            if diffshiftup >= 0:
                pano[diffshiftup:leftSize[0]+diffshiftup, :diff1x+w//2] = left_warp[:, :diff1x+w//2]
                pano[:rightSize[0], diff1x+w//2:] = right_warp[:, w//2:]
            else:
                min_height = min(leftSize[0], pano.shape[0])
                pano[:min_height, :diff1x+w//2] = left_warp[:min_height, :diff1x+w//2]
                pano[-diffshiftup:rightSize[0]-diffshiftup, diff1x+w//2:] = right_warp[:, w//2:]
        except:
            print("only right part")
            pano = right_warp
    else:
        print("Alignement problem")
        # If there are errors in the calculation due to bad image order or bad logical calculations
        pano = right_warp


    #Uncomment if clipping wanted 
    # max_width = int(1.5 * w)
    # clip_x_start = max(0, diff1x - max_width)  # Clip left side if needed
    # clip_x_end = min(pano.shape[1], diff1x + max_width + w)  # Clip right side if needed

    # # Clip panorama height if diff1y or diff2y exceed 1.5 times the height (h)
    # max_height = int(1.5 * h)
    # clip_y_start = max(0, top_shift - max_height)  # Clip top side based on top_shift
    # clip_y_end = min(pano.shape[0], bottom_shift + max_height)  # Clip bottom side based on bottom_shift

    # # Apply clipping to both width and height
    # pano_clipped = pano[clip_y_start:clip_y_end, clip_x_start:clip_x_end]
    # return pano_clipped
    return pano