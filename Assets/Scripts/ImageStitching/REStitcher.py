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
import yaml

import sys
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.abspath("Residual_Elastic_Warp_main"))
import Residual_Elastic_Warp_main.models
import Residual_Elastic_Warp_main.utils

import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import GaussianBlur

# from transformers import SuperPointForKeypointDetection
# # from torch.quantization import quantize_dynamic
# from numba import jit

already_saved = True

class REStitcher(BaseStitcher):
    def __init__(self):
        super().__init__(device="cpu")  # Initialize the base class

        # UDIS parameters
        self.resize_512 = T.Resize((512,512))
        yaml_path = "Residual_Elastic_Warp_main/configs/test/rewarp.yaml"

        with open(yaml_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            print('config loaded.')

        sv_file = torch.load("Residual_Elastic_Warp_main/pretrained/rewarp.pth")
        self.model = Residual_Elastic_Warp_main.models.make(sv_file['T_model'], load_sd=True)
        self.H_model = Residual_Elastic_Warp_main.models.make(sv_file['H_model'], load_sd=True)
    
    def RE_warping(self, image1, image2, test = True):
        ref , tgt, ref_, tgt_  = loadImages(image1, image2)

        hcell_iter = 6
        tcell_iter = 3
        self.model.iters_lev = tcell_iter
        
        b, c, h, w = ref_.shape
        scale = h/ref.shape[-2]

        _, disps, hinp = self.H_model(ref, tgt, iters_lev0=hcell_iter)

        t0 = time.time()
        # Preparation of warped inputs
        H, img_h, img_w, offset = Residual_Elastic_Warp_main.utils.get_warped_coords(disps[-1], scale=(h/512, w/512), size=(h,w))
        H_, *_ = Residual_Elastic_Warp_main.utils.get_H(disps[-1].reshape(ref.shape[0],2,-1).permute(0,2,1), [*ref.shape[-2:]])
        H_ = Residual_Elastic_Warp_main.utils.compens_H(H_, [*ref.shape[-2:]])

        grid = Residual_Elastic_Warp_main.utils.make_coordinate_grid([*ref.shape[-2:]], type=H_.type())
        grid = grid.reshape(1, -1, 2).repeat(ref.shape[0], 1, 1)

        mesh_homography = Residual_Elastic_Warp_main.utils.warp_coord(grid, H_.cuda()).reshape(b,*ref.shape[-2:],-1)
        ones = torch.ones_like(ref_).cuda()
        tgt_w = F.grid_sample(tgt, mesh_homography, align_corners=True)


        # Warp Field estimated by TPS
        flows = self.model(tgt_w, ref, iters=tcell_iter, scale=scale)
        translation = Residual_Elastic_Warp_main.utils.get_translation(*offset)
        T_ref = translation.clone()
        T_tgt = torch.inverse(H).double() @ translation.cuda()

        sizes = (img_h, img_w)
        if img_h > 5000 or img_w > 5000:
            print(sizes)
            print('Fail; Evaluated Size: {}X{}'.format(img_h, img_w))
            flows, disps= None, None
            return None, None


        # Image Alignment
        coord1 = Residual_Elastic_Warp_main.utils.to_pixel_samples(None, sizes=sizes).cuda()
        mesh_r, _ = Residual_Elastic_Warp_main.utils.gridy2gridx_homography(
            coord1.contiguous(), *sizes, *tgt_.shape[-2:], T_ref.cuda(), cpu=False
        )
        mesh_r = mesh_r.reshape(b, img_h, img_w, 2).cuda().flip(-1)

        coord2 = Residual_Elastic_Warp_main.utils.to_pixel_samples(None, sizes=sizes).cuda()
        mesh_t, _ = Residual_Elastic_Warp_main.utils.gridy2gridx_homography(
            coord2.contiguous(), *sizes, *tgt_.shape[-2:], T_tgt.cuda(), cpu=False
        )
        mesh_t = mesh_t.reshape(b, img_h, img_w, 2).cuda().flip(-1)

        mask_r = F.grid_sample(ones, mesh_r, mode='nearest', align_corners=True)
        mask_t = F.grid_sample(ones, mesh_t, mode='nearest', align_corners=True)

        # ovl = torch.round(mask_r * mask_t)

        flow = flows[-1]/511
        if flow.shape[-2] != 512 or flow.shape[-1] != 512:
            flow = F.interpolate(flow.permute(0,3,1,2), size=(h, w), mode='bilinear').permute(0,2,3,1) * 2

        mesh_t[:, offset[0]:offset[0]+h, offset[1]:offset[1]+w, :] += flow
        ref_w = F.grid_sample(ref_, mesh_r, mode='bilinear', align_corners=True)
        mask_r = F.grid_sample(ones, mesh_r, mode='nearest', align_corners=True)
        tgt_w = F.grid_sample(tgt_, mesh_t, mode='bilinear', align_corners=True)
        mask_t = F.grid_sample(ones, mesh_t, mode='nearest', align_corners=True)

        ref_w = (ref_w + 1)/2 * mask_r
        tgt_w = (tgt_w + 1)/2 * mask_t

        t1 = time.time()
        # print("Model inference", time.time()-t0)
        # Image Stitching
        stit = linear_blender(ref_w, tgt_w, mask_r, mask_t)
        
        # print("Linear blending", time.time()-t1)
        stit_ = stit[0].detach().cpu().numpy()*255.  # Convert to NumPy
        
        # stit_ = (stit_ + 1) / 2 * 255  # Scale from [-1, 1] to [0, 255]
        stit_ = stit_.clip(0, 255).astype('uint8')  # Clip to valid range

        # Transpose to [H, W, C] for saving
        stit_ = stit_.transpose(1, 2, 0)
        mask_r = mask_r.cpu().squeeze(0)

        torch.cuda.empty_cache()
        return stit_, mask_r.cpu().squeeze(0).numpy()[0]#.transpose(1, 2, 0)

    def RE_pano(self, images, subset1, subset2):
        global already_saved
        t0=time.time()
        h, w, _ = images[0].shape
        test= False

        
        right_warp, right_mask = self.RE_warping(images[subset2[0]], images[subset2[1]], test)
        t1 = time.time()
        # We have to flip the images to warp the left image and keep central image as the reference
        image1, image2 = cv2.flip(images[subset1[0]], 1), cv2.flip(images[subset1[1]], 1)
        left_warp, left_mask = self.RE_warping(image1, image2, test)
        t2 = time.time()
        
        if right_warp is None:
            if left_warp is None:
                return None
            else:
                return left_warp
        elif left_warp is None:
            return right_warp

        pano = ComposeTwoSides(left_warp, right_warp, left_mask, right_mask, size=(h, w))
        # pano = right_warp

        # print(f"Warp time : {time.time()-t0}")
        # print(f"First Warp time : {t1-t0}")
        # print(f"Second time : {t2-t1}")

        if not already_saved:
            already_saved = True
            cv2.imwrite("left_warp.jpg", cv2.cvtColor(left_warp.astype('uint8'), cv2.COLOR_RGB2BGR))
            cv2.imwrite("right_warp.jpg", cv2.cvtColor(right_warp.astype('uint8'), cv2.COLOR_RGB2BGR))
            cv2.imwrite("left_mask.jpg", (left_mask * 255).astype('uint8'))
            cv2.imwrite("right_mask.jpg", (right_mask * 255).astype('uint8'))
            cv2.imwrite("pano.jpg", cv2.cvtColor(pano.astype('uint8'), cv2.COLOR_RGB2BGR))
            print("saving")

        return pano.astype(np.uint8)
  
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
            pano = self.RE_pano(images, subset1, subset2)          
        if verbose:
            print(f"Warp time: {time.time()-t}")    
        return pano

@jit(nopython=True)
def preprocess_images(images):
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
    h, w = size
    rightSize = right_warp.shape
    leftSize = left_warp.shape

    shiftup1 = find_image_shift(left_mask[:, :w//4])
    shiftup2 = find_image_shift(right_mask[:, :w//4])
    shiftdown1 = max(leftSize[0]-h-shiftup1, 0)
    shiftdown2 = max(rightSize[0]-h-shiftup2, 0)

    left_warp = cv2.flip(left_warp, 1)

    # Compute difference size between ref image and warped ones
    diff2x, diff2y = rightSize[1]-w, rightSize[0]-h
    diff1x, diff1y = leftSize[1]-w, leftSize[0]-h

    top_shift = max(shiftup1, shiftup2)
    bottom_shift = max((leftSize[0] - shiftup1), (rightSize[0] - shiftup2))

    # pano = np.zeros((diff1y+diff2y+h, diff1x+diff2x+w, 3))
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

        condition1 = diff1y == shiftup1+shiftdown1
        condition2 = diff2y == shiftup2+shiftdown2

        if not condition1 and not condition2:
            print(f"diff1x: {diff1x}, diff1y: {diff1y}, shiftup1: {shiftup1}, shiftdown1: {shiftdown1}")
            print(f"diff2x: {diff2x}, diff2y: {diff2y}, shiftup2: {shiftup2}, shiftdown2: {shiftdown2}")
        elif not condition1:
            print(f"diff1x: {diff1x}, diff1y: {diff1y}, shiftup1: {shiftup1}, shiftdown1: {shiftdown1}")
        elif not condition2:
            print(f"diff2x: {diff2x}, diff2y: {diff2y}, shiftup2: {shiftup2}, shiftdown2: {shiftdown2}")
        
        # If there are errors in the calculation due to bad image order or bad logical calculations
        pano = right_warp

    return pano


def loadImages(ref, tgt):
    
    h, w, c = ref.shape
    if h != 512 or w != 512:
        ref_ = cv2.resize(ref, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        tgt_ = cv2.resize(tgt, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
    
    ref =  (ref/255. - 0.5) * 2
    tgt = (tgt/255. - 0.5) * 2
    img_ref = (ref_/255. - 0.5) * 2
    img_tgt = (tgt_/255. - 0.5) * 2

    img_ref = np.transpose(img_ref, (2,0,1)).astype(np.float32)
    img_tgt = np.transpose(img_tgt, (2,0,1)).astype(np.float32)
    ref = np.transpose(ref, (2,0,1)).astype(np.float32)
    tgt = np.transpose(tgt, (2,0,1)).astype(np.float32)

    return torch.from_numpy(img_ref).unsqueeze(0).cuda(), torch.from_numpy(img_tgt).unsqueeze(0).cuda(), torch.from_numpy(ref).unsqueeze(0).cuda(), torch.from_numpy(tgt).unsqueeze(0).cuda()

def linear_blender(ref, tgt, ref_m, tgt_m, mask=False):
    blur = GaussianBlur(kernel_size=(21,21), sigma=20)
    r1, c1 = torch.nonzero(ref_m[0, 0], as_tuple=True)
    r2, c2 = torch.nonzero(tgt_m[0, 0], as_tuple=True)

    center1 = (r1.float().mean(), c1.float().mean())
    center2 = (r2.float().mean(), c2.float().mean())

    vec = (center2[0] - center1[0], center2[1] - center1[1])

    ovl = (ref_m * tgt_m).round()[:, 0].unsqueeze(1)
    ref_m_ = ref_m[:, 0].unsqueeze(1) - ovl
    r, c = torch.nonzero(ovl[0, 0], as_tuple=True)

    ovl_mask = torch.zeros_like(ref_m_).cuda()
    proj_val = (r - center1[0]) * vec[0] + (c - center1[1]) * vec[1]
    ovl_mask[ovl.bool()] = (proj_val - proj_val.min()) / (proj_val.max() - proj_val.min() + 1e-3)

    mask1 = (blur(ref_m_ + (1-ovl_mask)*ref_m[:,0].unsqueeze(1)) * ref_m + ref_m_).clamp(0,1)
    if mask: return mask1

    mask2 = (1-mask1) * tgt_m
    stit = ref * mask1 + tgt * mask2

    return stit

# # To avoid too big images
#     diff2x, diff2y = min(diff2x, 3*w), min(diff2y, 2*h)
#     diff1x, diff1y = min(diff2x, 3*w), min(diff2y, 2*h)