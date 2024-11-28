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

import UDIS2_main.Warp.Codes.grid_res as grid_res
from UDIS2_main.Warp.Codes.network import build_output_model, get_stitched_result, Network, build_new_ft_model
from UDIS2_main.Warp.Codes.loss import cal_lp_loss2

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
        model_path ="UDIS2_main\Warp"
        MODEL_DIR = os.path.join(model_path, 'model')

        ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
        ckpt_list.sort()
        if len(ckpt_list) != 0:
            model_path = ckpt_list[-1]
            checkpoint = torch.load(model_path)
            self.net.load_state_dict(checkpoint['model'])
            print('load model from {}!'.format(model_path))
    
    def UDIS_warping(self, image1, image2):
        input1_tensor, input2_tensor = loadSingleData(image1, image2)

        if torch.cuda.is_available():
            input1_tensor = input1_tensor.cuda()
            input2_tensor = input2_tensor.cuda()

        input1_tensor_512 = self.resize_512(input1_tensor)
        input2_tensor_512 = self.resize_512(input2_tensor)

        with torch.no_grad():
            batch_out = build_new_ft_model(self.net, input1_tensor_512, input2_tensor_512)
        # warp_mesh = batch_out['warp_mesh']
        # warp_mesh_mask = batch_out['warp_mesh_mask']
        rigid_mesh = batch_out['rigid_mesh']
        mesh = batch_out['mesh']

        with torch.no_grad():
            output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)

        stitched_images = output['stitched'][0].cpu().detach().numpy().transpose(1,2,0)

        return stitched_images

    def UDIS_pano(self, images, subset1, subset2):

        t0=time.time()
        h, w, _ = images[0].shape
        # print("warping right")
        right_warp = self.UDIS_warping(images[subset2[0]], images[subset2[1]])

        # Find first black pixel and first black pixel after image in the first column
        sum_right = right_warp[:,0].sum(axis=1)
        shiftup2 = np.argmax(sum_right != 0)
        shiftdown2 = 0
        if sum_right[-1]==0:
            shiftdown2 = shiftup2 + np.argmax(sum_right[shiftup2:] == 0)

        # We have to flip the images to warp the left image and keep central image as the reference
        image1, image2 = cv2.flip(images[subset1[0]], 1), cv2.flip(images[subset1[1]], 1)
        left_warp = self.UDIS_warping(image1, image2)
        
        # Find first black pixel and first black pixel after image in the first column
        sum_left = left_warp[:,0].sum(axis=1)
        shiftup1 = np.argmax(sum_left != 0)
        shiftdown1 = 0
        if sum_left[-1]==0:
            shiftdown1 = shiftup1 + np.argmax(sum_left[shiftup1:] == 0)

        left_warp = cv2.flip(left_warp, 1)

        rightSize = right_warp.shape
        leftSize = left_warp.shape

        # Compute difference size between ref image and warped ones
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
                # Problem dimension here
                pano[-diffshiftup:rightSize[0]-diffshiftup, diff1x+w//2:] = right_warp[:, w//2:]
                # maybe this
                # pano[-diffshiftup:leftSize[0]-diffshiftup, diff1x+w//2:] = right_warp[:, w//2:]
                
        else:
            print("Control if shifts are equal:")
            print(diff2y==shiftup2+shiftdown2, diff1y==shiftup1+shiftdown1)
            
            print("Alignement problem")
            print(f"diff1x: {diff1x}, diff1y: {diff1y}, shiftup1: {shiftup1}, shiftdown1: {shiftdown1}")
            print(f"diff2x: {diff2x}, diff2y: {diff2y}, shiftup2: {shiftup2}, shiftdown2: {shiftdown2}")
            
            pano = right_warp

        print(f"Warp time : {time.time()-t0}")

        return pano.astype(np.uint8)

    def UDIS_batch_warping(self, image1, image2, image3):
        
        t0=time.time()
        image1, image2_flipped=cv2.flip(image1, 1), cv2.flip(image2, 1)
        input1_tensor, input2_tensor = load3images(image1, image2, image2_flipped, image3)
        t1=time.time()
        if torch.cuda.is_available():
            input1_tensor = input1_tensor.cuda()
            input2_tensor = input2_tensor.cuda()
        t2=time.time()
        input1_tensor_512 = self.resize_512(input1_tensor)
        input2_tensor_512 = self.resize_512(input2_tensor)
        t3=time.time()
        with torch.no_grad():
            batch_out = build_new_ft_model(self.net, input1_tensor_512, input2_tensor_512)
        rigid_mesh = batch_out['rigid_mesh']
        mesh = batch_out['mesh']
        t4=time.time()
        with torch.no_grad():
            output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)
        t5=time.time()
        stitched_images = output['stitched'].cpu().detach()#.numpy().transpose(1,2,0)
        t6=time.time()

        # print(f"loading time : {t1-t0}")
        # print(f"GPU transfer time : {t2-t1}")
        # print(f"Resize time : {t3-t2}")
        # print(f"Mesh computation time : {t4-t3}")
        # print(f"Stitching time : {t5-t4}")
        # print(f"CPU + detachement time : {t6-t5}")

        return stitched_images
    
    def UDIS_batch_pano(self, images, subset1, subset2):

        t0=time.time()
        h, w, _ = images[0].shape

        output = self.UDIS_batch_warping(images[subset1[1]], images[subset2[0]], images[subset2[1]])
        # input1_tensor, input2_tensor = load3images(images[subset1[1]], images[subset2[0]], images[subset2[1]])
        # out = build_output_model(self.net, input1_tensor.cuda(), input2_tensor.cuda())
        t1=time.time()

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

        t2=time.time()

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

        # print(f"UDIS time : {t1-t0}")
        # print(f"Placement calculation time : {t2-t1}")
        # print(f"If code time : {time.time()-t2}")
        # print(f"Warp time : {time.time()-t0}")

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
        pano = self.UDIS_pano(images, subset1, subset2)          
        if verbose:
            print(f"Warp time: {time.time()-t}")    
        return pano
    

def loadSingleData(image1, image2):

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