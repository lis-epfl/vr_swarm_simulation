import numpy as np
import cv2
import glob
import os
import sys
import torch
import time
from transformers import SuperPointForKeypointDetection
# from torch.quantization import quantize_dynamic
from numba import jit
import queue
import threading
import mmap
import struct
import networkx as nx
import random
from PIL import Image

from BaseStitcher import *

sys.path.append(os.path.abspath("UDIS2_main\Warp\Codes"))
import UDIS2_main.Warp.Codes.utils_udis as udis_utils
import UDIS2_main.Warp.Codes.utils_udis.torch_DLT as torch_DLT
import UDIS2_main.Warp.Codes.grid_res as grid_res
from UDIS2_main.Warp.Codes.network import build_output_model, get_stitched_result, Network, build_new_ft_model
from UDIS2_main.Warp.Codes.loss import cal_lp_loss2
from UDISStitcher import *

sys.path.append(os.path.abspath("Neural_Image_Stitching_main"))
import Neural_Image_Stitching_main.srwarp
import Neural_Image_Stitching_main.utils as nis_utils
from Neural_Image_Stitching_main.models.ihn import *
from Neural_Image_Stitching_main.models import *
from Neural_Image_Stitching_main import stitch
import Neural_Image_Stitching_main.pretrained
from NISStitcher import *

sys.path.append(os.path.abspath("Residual_Elastic_Warp_main"))
import Residual_Elastic_Warp_main.models
import Residual_Elastic_Warp_main.utils
from REStitcher import *

# Activate environnement
# cmd
# cd Assets\Scripts\ImageStitching 
# python SamplingStitcher.py


from datetime import datetime

class StitcherManager:
    def __init__(self, device ="cpu"):

        self.stitchers = {
            "CLASSIC": BaseStitcher(algorithm=1, trees=5, checks=50, ratio_thresh=0.7, score_threshold=0.05, device=device),
            "UDIS": UDISStitcher(),
            "NIS": NISStitcher(),  # Replace with your NISStitcher instance if implemented
            "REWARP": REStitcher(),
        }

        # Manually remove models of NIS from GPU because they load them directly on GPU
        self.stitchers["NIS"].model.cpu(), self.stitchers["NIS"].H_model.cpu()

        self.active_stitcher = self.stitchers["CLASSIC"]
        self.active_stitcher_type = "CLASSIC"
        self.active_matcher_type = "BF"
        self.device = device
        
        self.switching_lock1 = threading.Lock()
        self.switching_lock2 = threading.Lock()
        self.info_lock = threading.Lock()

        self.stitcherTypes = ["CLASSIC", "UDIS", "NIS", "REWARP"]
        self.cylidnricalWarp = False
        self.isRANSAC = False
        self.headAngle = 0
        self.shared_images = None
        self.shared_images_bool = None

        self.processedImageWidth = None
        self.processedImageHeight = None
        self.batchImageWidth = None
        self.batchImageHeight = None

        self.order_queue = queue.Queue(1)
        self.homography_queue = queue.Queue(1)
        self.direction_queue = queue.Queue(1)
        self.panoram_queue = queue.Queue(1)

    def set_stitcher(self, stitcher_type, onlyIHN):
        """
        Safely switch the active stitcher.
        Waits for both threads to finish their current work before switching.
        """

        # REmove devices from GPU
        if self.active_stitcher_type == "CLASSIC":
            self.active_stitcher.superpoint_model.cpu()
        elif self.active_stitcher_type == "UDIS":
            self.active_stitcher.net.cpu()
        elif self.active_stitcher_type == "NIS":
            self.active_stitcher.model.cpu(), self.active_stitcher.H_model.cpu()
        elif self.active_stitcher_type == "REWARP":
            self.active_stitcher.model.cpu(), self.active_stitcher.H_model.cpu()

        torch.cuda.empty_cache()
        self.active_stitcher = self.stitchers[stitcher_type]
        self.active_stitcher_type = stitcher_type
        print(f"Switched to {self.active_stitcher.__class__.__name__}")
        if self.active_stitcher_type == "CLASSIC":
            self.active_stitcher.superpoint_model.to(self.device)
        elif self.active_stitcher_type == "UDIS":
            self.active_stitcher.net.to(self.device)
        elif self.active_stitcher_type == "NIS":
            self.active_stitcher.model.to(self.device), self.active_stitcher.H_model.to(self.device)
            self.active_stitcher.onlyIHN = onlyIHN
        elif self.active_stitcher_type == "REWARP":
            self.active_stitcher.model.to(self.device), self.active_stitcher.H_model.to(self.device)
    
    def process_thread2(self, images, front_image_index,Hs, verbose, debug):
        """
        Thread 2 operation. Uses lock_thread2 for thread-safe access.
        """
        with self.switching_lock1:
            # if self.active_stitcher is not None:
            try:
                Hs, order, inverted = self.active_stitcher.findHomographyOrder(images, front_image_index, Hs, verbose, debug)
            except:
                return None
            self.order_queue.put(order)
            self.homography_queue.put(Hs)
            self.direction_queue.put(inverted)
            print(order)
            return Hs


    def process_thread3(self, images, order, Hs, inverted, num_pano_img=3, verbose= False):
        """
        Thread 3 operation. Uses lock_thread3 for thread-safe access.
        """

        if self.active_stitcher_type == "CLASSIC":
            pano = self.active_stitcher.stitch(images, order, Hs, inverted, self.headAngle, self.processedImageWidth, self.processedImageHeight, num_pano_img=num_pano_img, verbose=verbose)
        elif self.active_stitcher_type == "UDIS":
            pano = self.active_stitcher.stitch(images, order, Hs, inverted ,self.headAngle, num_pano_img=num_pano_img, verbose=verbose)
        elif self.active_stitcher_type == "NIS":
            # pano = self.active_stitcher.stitch(images, self.headAngle, order, num_pano_img=num_pano_img, verbose=verbose)
            pano = self.active_stitcher.stitch(images, order, Hs, inverted , self.headAngle, num_pano_img=num_pano_img, verbose=verbose)
            # pano = np.zeros((self.processedImageHeight, self.processedImageWidth, 3))
        elif self.active_stitcher_type == "REWARP":
            pano = self.active_stitcher.stitch(images, order, Hs, inverted , self.headAngle, num_pano_img=num_pano_img, verbose=verbose)
        
        return pano

def first_thread(absolute_path_save = "save_images/", save_time = 10.0, debug = False):
    """""
    This method read the images coming from the software. They have three shared memory files to store the images, the panorama and the metadatas.
    """""
    flagPosition = 0
    metadataSize = 20 + 64 + 1+ 64+ 1 + 4*4 +1 # 20 bytes for ints (5x4 bytes) + 64 bytes for string + 1 byte bool
    metadataMMF = mmap.mmap(-1, metadataSize, "MetadataSharedMemory")

    # Read first time metadata to initialize the memories:
    output = readMetadataMemory(metadataMMF)
    
    batchImageWidth, batchImageHeight, imageCount, processedImageWidth , processedImageHeight= output["Sizes"]
    
    imageSize, headANglePosition, batchDataPosition, _ = UpdateValues(batchImageWidth, batchImageHeight, imageCount, processedImageWidth, processedImageHeight)
    batchMMF = mmap.mmap(-1, batchDataPosition +  imageCount* imageSize, "BatchSharedMemory")

    while True:

        ### New part
        output = readMetadataMemory(metadataMMF)
        batchImageWidth, batchImageHeight, imageCount, processedImageWidth, processedImageHeight= output["Sizes"]
        imageSize, headANglePosition, batchDataPosition, _ = UpdateValues(batchImageWidth, batchImageHeight, imageCount, processedImageWidth,processedImageHeight)

        try:
            batchMMF = mmap.mmap(-1, batchDataPosition +  imageCount* imageSize, "BatchSharedMemory")
            images, images_bool, _ = readMemory(batchMMF, flagPosition, headANglePosition, imageCount, batchDataPosition, imageSize, batchImageWidth, batchImageHeight)
        except:
            print("problem opening/reading batched memory")
            continue

        SAVE_PATH = absolute_path_save + "("+ str(batchImageWidth)+ "_" + str(batchImageWidth)+ ")" + datetime.now().strftime("%m%d%y%H%M%S")+ '/'
        save_images(images, SAVE_PATH, images_bool)

        time.sleep(save_time)

        if debug:
            break


def second_thread(manager: StitcherManager, front_image_index=0, verbose = False, debug= False):
    """""
    This method uses some of the above methods to extract the order and the homographies of the paired images.
    Input:
        - images: list of NDArrays.
        - front_image_index: the index of the front image of the pilot
    """""
    Hs = None
    # global shared_images
    while True:
        if manager.shared_images is None:
            print("Second thread sleep")
            time.sleep(0.4)
            continue
        
        with manager.info_lock:
            images = manager.shared_images

        t = time.time()
        Hs = manager.process_thread2(images, front_image_index=front_image_index, Hs = Hs, verbose = verbose, debug= debug)

        if verbose:
            print(f"Second thread loop time: {time.time()-t}")
        # time.sleep(0.5)
        if debug:
            break
    
def third_thread(manager: StitcherManager, num_pano_img=3, verbose =False, debug=False):
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

    order, Hs, inverted = None, None, None

    while True:
        if not manager.homography_queue.empty():
            del order, Hs, inverted
            order = manager.order_queue.get()
            Hs = manager.homography_queue.get()
            inverted = manager.direction_queue.get()
            order = np.hstack((order, order, order))
            Hs = np.concatenate((Hs, Hs, Hs))
        elif order is None:
            time.sleep(0.4)
            continue
        
        with manager.info_lock:
            images = manager.shared_images

        t = time.time()

        if order.shape[0]//3 != len(images):
            order = None
            continue
        manager.process_thread3(images,order, Hs, inverted, num_pano_img=3, verbose=verbose)
        # order = None

        if verbose:
            print(f"Third thread loop time: {time.time()-t}")
        if debug:
            break

    print("Quitting third thread")

def readMetadataMemory(metadataMMF :mmap )->dict:
    
    # Read the integers for image sizes
    metadataMMF.seek(0)
    int_values = struct.unpack('iiiii', metadataMMF.read(20))  # 5 integers

    # Read the string for Stitcher Type
    raw_string = metadataMMF.read(64)
    metadata_string = raw_string.decode('utf-8').rstrip('\x00')  # Remove padding

    # Read the boolean isCylindrical
    raw_bool = metadataMMF.read(1)
    metadata_bool = bool(struct.unpack('B', raw_bool)[0])  # Unpack as unsigned char

    # Read the string for BF or FLANN
    raw_string = metadataMMF.read(64)
    matcherType = raw_string.decode('utf-8').rstrip('\x00')  # Remove padding

    # Read the boolean for RANSAC or not
    raw_bool = metadataMMF.read(1)
    isRANSAC = bool(struct.unpack('B', raw_bool)[0])  # Unpack as unsigned char

    # Read the integer for check sizes
    checks = struct.unpack('i', metadataMMF.read(4))[0]  # 4 bytes for int

    # Read the float for ratio and score
    floats = struct.unpack('ff', metadataMMF.read(8))  # 2 floats (4 bytes each)

    # Read the integer for check sizes
    focal = struct.unpack('i', metadataMMF.read(4))[0]  # 4 bytes for int

    # Read the boolean for RANSAC or not
    raw_bool = metadataMMF.read(1)
    onlyIHN = bool(struct.unpack('B', raw_bool)[0])  # Unpack as unsigned char

    # Return all parsed metadata
    return {
        "Sizes": int_values,  # Tuple of 5 integers
        "typeOfStitcher": metadata_string,
        "isCylindrical": metadata_bool,
        "matcherType" : matcherType,
        "isRANSAC" : isRANSAC,
        "checks" : checks,
        "ratio_thresh" : floats[0],
        "score_threshold" : floats[1],
        "focal" : focal,
        "onlyIHN" : onlyIHN,
    }

@jit(nopython=True) 
def UpdateValues(batchImageWidth, batchImageHeight, imageCount, processedImageWidth, processedImageHeight):
    
    imageSize = batchImageWidth*batchImageHeight*3
    headAnglePosition = 5
    batchDataPosition = headAnglePosition+ imageCount

    processedImageSize = processedImageWidth*processedImageHeight*3

    return imageSize, headAnglePosition, batchDataPosition, processedImageSize
    
 
def readMemory(batchMMF, batchFlagPosition, headANglePosition, imageCount, batchDataPosition, imageSize, imageWidth, imageHeight):
    """
        Read Memory shared with Unity code. If the flag is 0, we can access data.

        Input:
            - batchMMF: mmap object for the shared memory
            - batchFlagPosition: position of the flag in the memory. In our case: 0
            - imageCount: number of images
            - batchDataPosition: position of the first image in memory. In our case: 1 + imageCount (1 byte for the flag + imageCount bytes for the boolean)
            - imageSize: The full image size (imageHeight * imageWidth * 3)
            - imageWidth
            - imageHeight
    """
    
    images = []
    while True:
        # Read the flag to check if Unity has written new images
        batchMMF.seek(batchFlagPosition)
        flag = struct.unpack('B', batchMMF.read(1))[0]

        if flag == 0:  # Unity isn't writing new images
            
            # Flag to 1, indicating we are reading
            batchMMF.seek(batchFlagPosition)
            batchMMF.write(struct.pack('B', 1))

            batchMMF.seek(1)
            boolean_list = [bool(b) for b in batchMMF.read(imageCount)]
            boolean_array=np.array(boolean_list)

            batchMMF.seek(headANglePosition)
            headAngle = struct.unpack('f', batchMMF.read(4))[0]

            for i in range(imageCount):
                if not boolean_array[i]:
                    continue
                # Read each image sequentially from the shared memory
                batchMMF.seek(batchDataPosition + i * imageSize)
                image_data = batchMMF.read(imageSize)

                # Convert the byte array into a numpy array
                image = np.frombuffer(image_data, dtype=np.uint8)
                image = image.reshape((imageHeight, imageWidth, 3))  # Reshape to RGB format
                images.append(cv2.flip(image, 0))

            # Reset flag to 0, indicating we've read the images
            batchMMF.seek(batchFlagPosition)
            batchMMF.write(struct.pack('B', 0))

            return images, boolean_array, headAngle

def save_images(image_list, folder_path, bool_array):
    """
    Save images from a list into a folder with labels determined by a boolean array.

    Args:
        image_list (list): List of image arrays (e.g., numpy arrays from OpenCV).
        folder_path (str): Path to the folder where images will be saved.
        bool_array (list): Boolean array determining the labels of the images.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Initialize variables
    label = 0
    for i, image in enumerate(image_list):
        # Find the next valid label from the boolean array
        while label < len(bool_array) and not bool_array[label]:
            label += 1

        # If no more valid labels, break the loop
        if label >= len(bool_array):
            print(f"No more valid labels for image {i}. Stopping.")
            break

        # Define the filename with the label
        filename = os.path.join(folder_path, f"image_{label}.png")
        
        # Save the image
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Saved image {i} as {filename}")

        # Increment label
        label += 1


def stitch_saved_images(save_path, device = "cuda"):

    # stitcher = BaseStitcher(algorithm=1, trees=5, checks=50, ratio_thresh=0.7, score_threshold=0.05, device=device)
    # for folderename in os.listdir(save_path):
    #     folderpath = os.path.join(save_path, folderename)
    #     stitch_folder(stitcher, folderpath, stitcher_type = "CLASSIC")
    #     # # break
    
    stitcher = None
    stitcher = UDISStitcher()
    stitcher.superpoint_model.to(device)
    stitcher.net.to(device)

    for folderename in os.listdir(save_path):
        folderpath = os.path.join(save_path, folderename)
        stitch_folder(stitcher, folderpath, stitcher_type = "UDIS_GIVEN_ORDER")

    stitcher = None
    stitcher = NISStitcher()
    stitcher.superpoint_model.to(device)
    stitcher.model.to(device), stitcher.H_model.to(device)

    for folderename in os.listdir(save_path):
        folderpath = os.path.join(save_path, folderename)
        stitch_folder(stitcher, folderpath, stitcher_type = "NIS_GIVEN_ORDER")

    stitcher = None
    stitcher = REStitcher()
    stitcher.superpoint_model.to(device)
    stitcher.model.to(device), stitcher.H_model.to(device)

    for folderename in os.listdir(save_path):
        folderpath = os.path.join(save_path, folderename)
        stitch_folder(stitcher, folderpath, stitcher_type = "REWARP_GIVEN_ORDER")

def stitch_folder(stitcher, folderpath, stitcher_type = "CLASSIC"):
    """
    Takes folderpath, load images in the folder and stitch the images
    """
    images = []

    # Load all images from the folder
    for filename in sorted(os.listdir(folderpath), key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else float('inf')):
        filepath = os.path.join(folderpath, filename)
        if os.path.isfile(filepath):
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)  # Load as RGB
            if image is not None:
                images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to RGB
            # print(filepath)
    h, w, c = images[0].shape
    
    _, Hs, order, inverted, _, _, confidences = stitcher.findHomographyOrder(images, 0, None, verbose = False, debug= True)
    print(order)
    order = [0, 1, 2, 3, 5, 7, 11, 10, 9, 8, 6, 4]
    order = np.hstack((order, order, order))
    Hs = np.concatenate((Hs, Hs, Hs))
    
    num_images = len(images)
    angle_per_image = 360/num_images
    processedImageWidth, processedImageHeight = w*4, h*2
    folderpath = os.path.join(folderpath, stitcher_type)
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    for i in range(num_images):
        headAngle = i*angle_per_image
        if stitcher_type == "CLASSIC":
            pano = stitcher.stitch(images, order, Hs, inverted, headAngle, processedImageWidth, processedImageHeight, num_pano_img=3, verbose = False)
        elif stitcher_type == "UDIS_GIVEN_ORDER":
            pano = stitcher.stitch(images, order, Hs, inverted , headAngle, num_pano_img=3, verbose = False)
        elif stitcher_type == "NIS_GIVEN_ORDER":
            pano = stitcher.stitch(images, order, Hs, inverted , headAngle, num_pano_img=3, verbose = False)
        elif stitcher_type == "REWARP_GIVEN_ORDER":
            pano = stitcher.stitch(images, order, Hs, inverted , headAngle, num_pano_img=3, verbose = False)
        
        # Define the filename with the label
        filename = os.path.join(folderpath, f"angle_{int(headAngle)}_{stitcher_type}.png")
        
        # # Save the image
        cv2.imwrite(filename, cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
        # print(f"Saved image {i} as {filename}")
        
def main():

    SAVING = False

    imageWidth = 300
    imageHeight = 300

    f = 160

    cam_mat = np.array([[f, 0, imageWidth/2], 
                        [0, f, imageHeight/2],
                        [0, 0, 1]])
    
    manager = StitcherManager("cuda")
    verbose = False
    debug = False
    absolute_path_save = "C:/Users/guill/OneDrive/Bureau/image_samples/"
    save_time = 50.0

    if SAVING:
        first_t = threading.Thread(target=first_thread ,args=(absolute_path_save, save_time, debug))
        first_t.daemon = True
        first_t.start()
        while True:
            time.sleep(100)
    else:
        stitch_saved_images(absolute_path_save, device = "cuda")
    

if __name__ == '__main__':
    
    # main function, the final result
    main()