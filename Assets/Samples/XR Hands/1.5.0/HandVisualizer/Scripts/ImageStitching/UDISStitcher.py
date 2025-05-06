# import custom_stitching
import numpy as np
import cv2
import torch
import time
from BaseStitcher import *


import glob
import os
import torch
import time
from numba import jit

import sys
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.abspath("UDIS2_main\Warp\Codes"))

from UDIS2_main.Warp.Codes.utils_udis import *

# import UDIS2_main.Warp.Codes.grid_res as grid_res
from UDIS2_main.Warp.Codes.network import build_output_model, get_stitched_result, Network, build_new_ft_model
# from UDIS2_main.Warp.Codes.loss import cal_lp_loss2

import torchvision.transforms as T

# from transformers import SuperPointForKeypointDetection
# # from torch.quantization import quantize_dynamic
# from numba import jit

class UDISStitcher(BaseStitcher):
    def __init__(self):
        super().__init__(device="cpu")  # Initialize the base class

        # UDIS parameters
        self.resize_512 = T.Resize((512,512))
        self.net = Network()
        self.net.eval()
        model_path ="UDIS2_main\Warp"
        MODEL_DIR = os.path.join(model_path, 'model')

        ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
        ckpt_list.sort()
        if len(ckpt_list) != 0:
            model_path = ckpt_list[-1]
            checkpoint = torch.load(model_path)
            self.net.load_state_dict(checkpoint['model'])
    
    def UDIS_warping(self, image1, image2, test = True):
        """
        Performs warping and stitching of two images using the UDIS (Unsupervised Deep Image Stitching) model.

        Parameters:
        - image1 (np.ndarray): The first input image.
        - image2 (np.ndarray): The second input image.
        - test (bool): A flag for enabling test-specific behavior. Default is True.

        Returns:
        - stitched_images (np.ndarray): The stitched image resulting from the UDIS model.
        - final_warp1_mask (np.ndarray): The mask for the first warped image.
        """
        input1_tensor, input2_tensor = loadSingleData(image1, image2)
        
        # ### More comuptationally intensive
        # if torch.cuda.is_available():
        #     input1_tensor = input1_tensor.cuda()
        #     input2_tensor = input2_tensor.cuda()

        # input1_tensor_512 = self.resize_512(input1_tensor)
        # input2_tensor_512 = self.resize_512(input2_tensor)

        # with torch.no_grad():
        #     batch_out = build_new_ft_model(self.net, input1_tensor_512, input2_tensor_512)
        # rigid_mesh = batch_out['rigid_mesh']
        # mesh = batch_out['mesh']

        # with torch.no_grad():
        #     output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)

        # stitched_images = output['stitched'][0].cpu().detach().numpy().transpose(1,2,0)
        # return stitched_images, 0

        if torch.cuda.is_available():
            inpu1_tesnor = input1_tensor.cuda()
            inpu2_tesnor = input2_tensor.cuda()

        with torch.no_grad():
            batch_out = build_output_model(self.net, inpu1_tesnor, inpu2_tesnor)

        final_warp1 = batch_out['final_warp1']
        final_warp1_mask = batch_out['final_warp1_mask']
        final_warp2 = batch_out['final_warp2']
        final_warp2_mask = batch_out['final_warp2_mask']

        final_warp1 = ((final_warp1[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        final_warp2 = ((final_warp2[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        final_warp1_mask = final_warp1_mask[0].cpu().detach().numpy().transpose(1,2,0)
        final_warp2_mask = final_warp2_mask[0].cpu().detach().numpy().transpose(1,2,0)

        sum_warp = final_warp1+final_warp2+1e-6
        stitched_images = final_warp1 * (final_warp1/ sum_warp) + final_warp2 * (final_warp2/sum_warp)
        batch_out = None
        return stitched_images, final_warp1_mask

        

    def UDIS_pano(self, images, subset1, subset2):
        """
        Creates a panorama by stitching images using the UDIS (Unsupervised Deep Image Stitching) model.

        Parameters:
        - images (list): List of input images to be stitched into a panorama.
        - subset1 (np.ndarray): Indices of images on the left side of the panorama.
        - subset2 (np.ndarray): Indices of images on the right side of the panorama.

        Returns:
        - pano (np.ndarray): The final stitched panorama image in uint8 format.
        """
        h, w, _ = images[0].shape
        test= True

        right_warp, right_mask = self.UDIS_warping(images[subset2[0]], images[subset2[1]], test)

        # We have to flip the images to warp the left image and keep central image as the reference
        image1, image2 = cv2.flip(images[subset1[0]], 1), cv2.flip(images[subset1[1]], 1)
        left_warp, left_mask = self.UDIS_warping(image1, image2, test)
        pano = ComposeTwoSides(left_warp, right_warp, left_mask, right_mask, size=(h, w))

        return pano.astype(np.uint8)

    def UDIS_batch_warping(self, image1, image2, image3):
        """
        Performs batch warping and stitching of three images using the UDIS (Unsupervised Deep Image Stitching) model.

        Parameters:
        - image1 (np.ndarray): The first input image. Left image
        - image2 (np.ndarray): The second input image. Middle image
        - image3 (np.ndarray): The third input image. Right image

        Returns:
        - stitched_images (torch.Tensor): The stitched batch of images as a PyTorch tensor.
        """
        
        image1, image2_flipped=cv2.flip(image1, 1), cv2.flip(image2, 1)
        input1_tensor, input2_tensor = load3images(image1, image2, image2_flipped, image3)
        if torch.cuda.is_available():
            input1_tensor = input1_tensor.cuda()
            input2_tensor = input2_tensor.cuda()
        input1_tensor_512 = self.resize_512(input1_tensor)
        input2_tensor_512 = self.resize_512(input2_tensor)
        with torch.no_grad():
            batch_out = build_new_ft_model(self.net, input1_tensor_512, input2_tensor_512)
        rigid_mesh = batch_out['rigid_mesh']
        mesh = batch_out['mesh']
        with torch.no_grad():
            output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)
        stitched_images = output['stitched'].cpu().detach()#.numpy().transpose(1,2,0)

        return stitched_images
    
    def UDIS_batch_pano(self, images, subset1, subset2):
        """
        Creates a panorama from a batch of images using the UDIS (Unsupervised Deep Image Stitching) model.

        Parameters:
        - images (list): List of input images to be stitched into a panorama.
        - subset1 (np.ndarray): Indices of images on the left side of the panorama.
        - subset2 (np.ndarray): Indices of images on the right side of the panorama.

        Returns:
        - pano (np.ndarray): The final stitched panorama image in uint8 format.
        """
        h, w, _ = images[0].shape

        output = self.UDIS_batch_warping(images[subset1[1]], images[subset2[0]], images[subset2[1]])
        # input1_tensor, input2_tensor = load3images(images[subset1[1]], images[subset2[0]], images[subset2[1]])
        # out = build_output_model(self.net, input1_tensor.cuda(), input2_tensor.cuda())

        left_warp, right_warp = output[0].numpy().transpose(1,2,0), output[1].numpy().transpose(1,2,0)
        
        shiftup1 = np.argmax(left_warp[:,0].sum(axis=1) != 0)
        shiftdown1 = shiftup1 + np.argmax(left_warp[shiftup1:, 0].sum(axis=1) == 0)
        shiftup2 = np.argmax(right_warp[:,0].sum(axis=1) != 0)
        shiftdown2 = shiftup2 + np.argmax(right_warp[shiftup2:, 0].sum(axis=1) == 0)
        

        left_warp = cv2.flip(left_warp, 1)

        rightSize = right_warp.shape
        leftSize = left_warp.shape

        diff2x, diff2y = rightSize[1]-w, rightSize[0]-h
        diff1x, diff1y = leftSize[1]-w, leftSize[0]-h

        pano = np.zeros((diff1y+diff2y+h, diff1x+diff2x+w, 3))
        # pano = np.zeros((h, diff1x+diff2x+w, 3))

        if diff2y == shiftup2+(rightSize[0]-shiftdown2) and diff1y == shiftup1+(leftSize[0]-shiftdown1):
            diffshiftup = shiftup2-shiftup1

            if diffshiftup >= 0:
                pano[diffshiftup:leftSize[0]+diffshiftup, :diff1x+w//2] = left_warp[:, :diff1x+w//2]
                pano[:rightSize[0], diff1x+w//2:] = right_warp[:, w//2:]
            else:
                pano[:leftSize[0], :diff1x+w//2] = left_warp[:, :diff1x+w//2]
                pano[-diffshiftup:rightSize[0]-diffshiftup, diff1x+w//2:] = right_warp[:, w//2:]

        else:
            print("Alignement problem")
        return pano.astype(np.uint8)
  
    def stitch(self, images, order, Hs, inverted, headAngle, num_pano_img=3):
        """""
        This method uses some of the above methods to stitch a part of the given images based on a criterion that could be the orientation
        of the pilots head and the desired number of images in the panorama.
        Input:
            - images: list of NDArrays.
            - angle : orientation of the pilots head (in degrees [0,360[?)
            - num_pano_img : desired number of images in the panorama
        """""
        subset1, subset2, _, _ = self.chooseSubsetsAndTransforms(Hs, num_pano_img, order, headAngle)
        pano = self.UDIS_pano(images, subset1, subset2)          
        
        return pano
    

def loadSingleData(image1, image2):
    """
    Prepares and preprocesses two input images for deep learning models.

    Parameters:
    - image1 (np.ndarray): The first input image with shape (height, width, channels).
    - image2 (np.ndarray): The second input image with shape (height, width, channels).

    Returns:
    - tuple: A tuple containing:
        - input1_tensor (torch.Tensor): The preprocessed tensor for the first image with shape (1, channels, height, width).
        - input2_tensor (torch.Tensor): The preprocessed tensor for the second image with shape (1, channels, height, width).
    """

    # load image1
    input1 = image1.astype(dtype=np.float32)
    input1 = (input1 / 127.5) - 1.0
    input1 = np.transpose(input1, [2, 0, 1])

    # load image2
    input2 = image2.astype(dtype=np.float32)
    input2 = (input2 / 127.5) - 1.0
    input2 = np.transpose(input2, [2, 0, 1])

    # convert to tensor
    input1_tensor = torch.tensor(input1).unsqueeze(0)
    input2_tensor = torch.tensor(input2).unsqueeze(0)
    return (input1_tensor, input2_tensor)

@jit(nopython=True)
def preprocess_images(images):
    """
    Preprocesses a list of images for deep learning models.

    Parameters:
    - images (list): A list of images in numpy array format, each with shape (height, width, channels).

    Returns:
    - processed_images (list): A list of preprocessed images, each with shape (channels, height, width),
                               normalized to the range [-1, 1].
    """
    processed_images = []
    for img in images:
        img = img.astype(np.float32)  # Convert to float32
        img = (img / 127.5) - 1.0     # Normalize to [-1, 1]
        img = img.transpose(2, 0, 1)  # Convert to channel-first format
        processed_images.append(img)
    return processed_images

def load3images(image1, image2, image2_flipped, image3):
    """""
    image1 : left image in panorama
    image2 : middle image in panorama
    image3 : right image in panorama

    """""
    images = [image1, image2, image2_flipped, image3]
    processed_images = preprocess_images(images)
    input1_tensor = torch.from_numpy(processed_images[0]).float()
    input2_tensor = torch.from_numpy(processed_images[1]).float()
    input2_flipped_tensor = torch.from_numpy(processed_images[2]).float()
    input3_tensor = torch.from_numpy(processed_images[3]).float()

    batch = (
        torch.stack((input2_flipped_tensor, input2_tensor)), 
        torch.stack((input1_tensor, input3_tensor))
    )
    return batch

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

    shiftup1 = find_image_shift(left_mask[:, 0])
    shiftup2 = find_image_shift(right_mask[:, 0])
    shiftdown1 = max(leftSize[0]-h-shiftup1, 0)
    shiftdown2 = max(rightSize[0]-h-shiftup2, 0)
    
    left_warp = cv2.flip(left_warp, 1)

    # Compute difference size between ref image and warped ones
    diff2x, diff2y = rightSize[1]-w, rightSize[0]-h
    diff1x, diff1y = leftSize[1]-w, leftSize[0]-h
    
    top_shift = max(shiftup1, shiftup2)
    bottom_shift = max((leftSize[0] - shiftup1), (rightSize[0] - shiftup2))
    pano = np.zeros((top_shift + bottom_shift+security_factor, diff1x+diff2x+w, 3))

    # pano = np.zeros((diff1y+diff2y+h, diff1x+diff2x+w, 3))

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
        pano = right_warp
        
        # print(diff2y==shiftup2+shiftdown2, diff1y==shiftup1+shiftdown1)
        # condition1 = diff1y == shiftup1+shiftdown1
        # condition2 = diff2y == shiftup2+shiftdown2
        
        # if not condition1 and not condition2:
        #     print(f"diff1x: {diff1x}, diff1y: {diff1y}, shiftup1: {shiftup1}, shiftdown1: {shiftdown1}")
        #     print(f"diff2x: {diff2x}, diff2y: {diff2y}, shiftup2: {shiftup2}, shiftdown2: {shiftdown2}")
        # elif not condition1:
        #     print(f"diff1x: {diff1x}, diff1y: {diff1y}, shiftup1: {shiftup1}, shiftdown1: {shiftdown1}")
        # elif not condition2:
        #     print(f"diff2x: {diff2x}, diff2y: {diff2y}, shiftup2: {shiftup2}, shiftdown2: {shiftdown2}")
        
        # If there are errors in the calculation due to bad image order or bad logical calculations
        

    max_width = int(1.5 * w)
    clip_x_start = max(0, diff1x - max_width)  # Clip left side if needed
    clip_x_end = min(pano.shape[1], diff1x + max_width + w)  # Clip right side if needed

    # Clip panorama height if diff1y or diff2y exceed 1.5 times the height (h)
    max_height = int(1.5 * h)
    clip_y_start = max(0, top_shift - max_height)  # Clip top side based on top_shift
    clip_y_end = min(pano.shape[0], bottom_shift + max_height)  # Clip bottom side based on bottom_shift

    # Apply clipping to both width and height
    pano_clipped = pano[clip_y_start:clip_y_end, clip_x_start:clip_x_end]

    return pano_clipped