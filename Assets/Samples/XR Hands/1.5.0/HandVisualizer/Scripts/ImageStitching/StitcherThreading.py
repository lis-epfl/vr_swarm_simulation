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
# python StitcherThreading.py

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

    def checkHyperparaChanges(self, output : dict):
        """
        Checks for changes in stitching type or hyperparameters and updates them if necessary.

        Parameters:
        - output (dict): A dictionary containing stitching settings and hyperparameters:
            - "typeOfStitcher" (str): The type of stitcher to use.
            - "isCylindrical" (bool): Whether cylindrical warping is enabled.
            - "matcherType" (str): The type of matcher to use (e.g., BF or FLANN).
            - "isRANSAC" (bool): Whether RANSAC is enabled.
            - "checks" (int): Number of checks for FLANN-based matching.
            - "ratio_thresh" (float): Lowe's ratio threshold for filtering matches.
            - "score_threshold" (float): Threshold for selecting keypoints based on their score.
            - "focal" (float): Focal length for cylindrical warping.
            - "onlyIHN" (bool): Whether to use only the IHN model.
            - "Sizes" (tuple): A tuple (batchImageWidth, batchImageHeight) specifying the batch image dimensions.
            """
        
        typeOfStitcher, isCylindrical, matcherType, isRANSAC  = output["typeOfStitcher"], output["isCylindrical"], output["matcherType"], output["isRANSAC"]
        checks, ratio_thresh, score_threshold, focal, onlyIHN = output["checks"], output["ratio_thresh"], output["score_threshold"], output["focal"], output["onlyIHN"]
        batchImageWidth, batchImageHeight = output["Sizes"][:2]

        def has_stitcher_changes():
            return (
                self.active_stitcher_type != typeOfStitcher and typeOfStitcher in self.stitcherTypes
            ) or (
                self.active_stitcher.cylindricalWarp != isCylindrical
            )

        def has_hyperparameter_changes():
            return (
                self.active_stitcher.active_matcher_type != matcherType or
                self.active_stitcher.isRANSAC != isRANSAC or
                self.active_stitcher.checks != checks or
                self.active_stitcher.ratio_thresh != ratio_thresh or
                self.active_stitcher.score_threshold != score_threshold or
                self.active_stitcher.focal != focal or
                self.batchImageWidth != batchImageWidth or
                self.batchImageHeight != batchImageHeight
            )

        
        if has_stitcher_changes():
            with self.switching_lock1:
                with self.switching_lock2:
                    self.set_stitcher(typeOfStitcher, onlyIHN)
                    self.changeCylindrical(isCylindrical)
                    self.changeCalculationsHyperpara(output)
                    pass
        if has_hyperparameter_changes():
            with self.switching_lock1:
                self.changeCalculationsHyperpara(output)
        
        if self.active_stitcher_type == "NIS" and self.active_stitcher.onlyIHN != onlyIHN:
            with self.switching_lock2:
                self.active_stitcher.onlyIHN = onlyIHN
    
    def process_thread2(self, images, front_image_index,Hs, verbose, debug):
        """
        Thread 2 operation. Uses lock_thread2 for thread-safe access. Extracts the homographies and order and put them in queues.
        """
        with self.switching_lock1:
            try:
                Hs, order, inverted = self.active_stitcher.findHomographyOrder(images, front_image_index, Hs, verbose, debug)
                # print(order)
            except:
                return None
            self.order_queue.put(order)
            self.homography_queue.put(Hs)
            self.direction_queue.put(inverted)
            return Hs

    def process_thread3(self, images, order, Hs, inverted, num_pano_img=3):
        """
        Thread 3 operation. Uses lock_thread3 for thread-safe access. Stitch the images by using information from the queues
        """
        with self.switching_lock2:
            # if self.active_stitcher:
            if self.active_stitcher_type == "CLASSIC":
                pano = self.active_stitcher.stitch(images, order, Hs, inverted, self.headAngle, self.processedImageWidth, self.processedImageHeight, num_pano_img=num_pano_img)
            elif self.active_stitcher_type == "UDIS":
                pano = self.active_stitcher.stitch(images, order, Hs, inverted ,self.headAngle, num_pano_img=num_pano_img)
            elif self.active_stitcher_type == "NIS":
                pano = self.active_stitcher.stitch(images, order, Hs, inverted , self.headAngle, num_pano_img=num_pano_img)
            elif self.active_stitcher_type == "REWARP":
                pano = self.active_stitcher.stitch(images, order, Hs, inverted , self.headAngle, num_pano_img=num_pano_img)
            if self.panoram_queue.empty():
                self.panoram_queue.put(pano)

    def changeCylindrical(self, isCylindrical):
        
        self.active_stitcher.cylindricalWarp = isCylindrical
        self.active_stitcher.points_remap = None
        pass

    def changeCalculationsHyperpara(self, output):
        self.active_stitcher.active_matcher_type = output["matcherType"]
        self.active_stitcher.isRANSAC = output["isRANSAC"]
        self.active_stitcher.checks = output["checks"]
        self.active_stitcher.search_params = dict(checks=output["checks"])
        self.active_stitcher.ratio_thresh = output["ratio_thresh"]
        self.active_stitcher.score_threshold = output["score_threshold"]
        focal = output["focal"]
        self.active_stitcher.focal = focal
        self.active_stitcher.camera_matrix = np.array([[focal,0, 150], [0,focal, 150], [0,0, 1]])
        self.batchImageWidth, self.batchImageHeight = output["Sizes"][:2]
        self.active_stitcher.points_remap = None
        pass

def first_thread(manager: StitcherManager, debug = False):
    """""
    This method read the images coming from the software. They have three shared memory files to store the images, the panorama and the metadatas.
    """""
    flagPosition = 0
    processedDataPosition = 4
    metadataSize = 20 + 64 + 1+ 64+ 1 + 4*4 +1 # 20 bytes for ints (5x4 bytes) + 64 bytes for string + 1 byte bool
    metadataMMF = mmap.mmap(-1, metadataSize, "MetadataSharedMemory")

    # Read first time metadata to initialize the memories:
    output = readMetadataMemory(metadataMMF)
    
    batchImageWidth, batchImageHeight, imageCount, manager.processedImageWidth , manager.processedImageHeight= output["Sizes"]
    
    imageSize, headANglePosition, batchDataPosition, processedImageSize = UpdateValues(batchImageWidth, batchImageHeight, imageCount, manager.processedImageWidth, manager.processedImageHeight)
    batchMMF = mmap.mmap(-1, batchDataPosition +  imageCount* imageSize, "BatchSharedMemory")
    processedMMF = mmap.mmap(-1, processedDataPosition + processedImageSize, "ProcessedImageSharedMemory")

    first_loop = True
    while True:

        ### New part
        output = readMetadataMemory(metadataMMF)
        batchImageWidth, batchImageHeight, imageCount, manager.processedImageWidth, manager.processedImageHeight= output["Sizes"]
        imageSize, headANglePosition, batchDataPosition, processedImageSize = UpdateValues(batchImageWidth, batchImageHeight, imageCount, manager.processedImageWidth, manager.processedImageHeight)
        manager.checkHyperparaChanges(output)

        try:
            batchMMF = mmap.mmap(-1, batchDataPosition +  imageCount* imageSize, "BatchSharedMemory")
            images, images_bool, headAngle = readMemory(batchMMF, flagPosition, headANglePosition, imageCount, batchDataPosition, imageSize, batchImageWidth, batchImageHeight)
        except:
            print("problem opening/reading batched memory")
            continue
        
        # droneImInd = np.arange(0, images_bool.shape[0])[images_bool]
        # print(f"Index of the drone images to stitch: {droneImInd}")
        with manager.info_lock:
            manager.shared_images = images
            manager.shared_images_bool = images_bool
            manager.headAngle = headAngle

        if not manager.panoram_queue.empty():
            panorama = manager.panoram_queue.get()
            H, W, _ = panorama.shape
            if H != manager.processedImageHeight or W != manager.processedImageWidth:
                try:
                    panorama = cv2.resize(panorama, (manager.processedImageWidth, manager.processedImageHeight))
                except:
                    continue
            try:
                processedMMF = mmap.mmap(-1, processedDataPosition + processedImageSize, "ProcessedImageSharedMemory")
                write_memory(processedMMF, flagPosition, processedDataPosition, processedImageSize, cv2.flip(panorama, 0))
                del panorama
            except:
                print("problem opening/reading processed memory")
                continue


        time.sleep(0.05)

        # print(headAngle, typeOfStitcher, isCylindrical)

        if first_loop:
            first_loop = False
            time.sleep(1.)

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
        manager.process_thread3(images,order, Hs, inverted, num_pano_img=num_pano_img)
        # order = None

        if verbose:
            print(f"Third thread loop time: {time.time()-t}")
        if debug:
            break

    print("Quitting third thread")

def readMetadataMemory(metadataMMF :mmap )->dict:
    """
    Reads metadata from a memory-mapped file and returns it as a dictionary.

    Parameters:
    - metadataMMF (mmap): The memory-mapped file object containing metadata.

    Returns:
    - dict: A dictionary containing the parsed metadata with the following keys:
        - "Sizes" (tuple): A tuple of 5 integers representing image dimensions and sizes.
        - "typeOfStitcher" (str): The type of stitcher used.
        - "isCylindrical" (bool): Whether cylindrical warping is enabled.
        - "matcherType" (str): The type of matcher used (e.g., BF or FLANN).
        - "isRANSAC" (bool): Whether RANSAC is enabled.
        - "checks" (int): The number of checks for FLANN-based matching.
        - "ratio_thresh" (float): Lowe's ratio threshold for filtering matches.
        - "score_threshold" (float): Threshold for selecting keypoints based on their score.
        - "focal" (int): Focal length for cylindrical warping.
        - "onlyIHN" (bool): Whether to use only the IHN model.
    """
    
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
    """
    Calculates and updates various values based on input image dimensions and counts.

    Parameters:
    - batchImageWidth (int): Width of the batch image.
    - batchImageHeight (int): Height of the batch image.
    - imageCount (int): Total number of images in the batch.
    - processedImageWidth (int): Width of the processed image.
    - processedImageHeight (int): Height of the processed image.

    Returns:
    - tuple: A tuple containing:
        - imageSize (int): Size of a batch image in bytes (batchImageWidth * batchImageHeight * 3 for RGB).
        - headAnglePosition (int): The starting position for head angle data (fixed at 5).
        - batchDataPosition (int): The starting position for batch data (headAnglePosition + imageCount).
        - processedImageSize (int): Size of a processed image in bytes (processedImageWidth * processedImageHeight * 3 for RGB).
    """
    
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

def write_memory(processedMMF, processedFlagPosition, processedDataPosition, processedImageSize, image_data):
    """
    Write an image to shared memory with Unity.

    Inputs:
        - processedMMF: mmap object for the shared memory.
        - processedFlagPosition: position of the flag in the memory
        - processedDataPosition: position to start writing the image data.
        - processedImageSize: expected size of the image data.
        - image_data: numpy array of the image to write.
    """
    while True:
        # Read the flag to check if Unity is ready for new data
        processedMMF.seek(processedFlagPosition)
        flag = struct.unpack('i', processedMMF.read(4))[0]

        if flag == 0:  # Unity isn't writing new images
            # Set flag to 1, indicating we're writing
            processedMMF.seek(processedFlagPosition)
            processedMMF.write(struct.pack('i', 1))

            # Convert image to byte array and check size
            image_bytes = image_data.tobytes()
            if len(image_bytes) != processedImageSize:
                raise ValueError(f"Image size mismatch: expected {processedImageSize}, got {len(image_bytes)}")

            # Write the image bytes to shared memory
            processedMMF.seek(processedDataPosition)
            processedMMF.write(image_bytes)

            # Reset flag to 0, indicating we've written the image
            processedMMF.seek(processedFlagPosition)
            processedMMF.write(struct.pack('i', 0))
            break

def main():
    """
    Activates the three threads and initialize the STitcherManager.
    """

    imageWidth = 300
    imageHeight = 300

    f = 160

    cam_mat = np.array([[f, 0, imageWidth/2], 
                        [0, f, imageHeight/2],
                        [0, 0, 1]])
    
    manager = StitcherManager("cuda")
    verbose_second_thread = False
    verbose_thrid_thread = True
    debug = False

    # To test only stitch time when order is known
    for key in manager.stitchers.keys():
        manager.stitchers[key].known_order = [ 0 , 1 , 2 , 3 , 5 , 7 , 11,  10 , 9 , 8 , 6 , 4] 

    first_t = threading.Thread(target=first_thread, args=(manager, debug))
    first_t.daemon = True
    first_t.start()

    sec_t = threading.Thread(target=second_thread, args=(manager, 0, verbose_second_thread, debug))
    sec_t.daemon = True
    sec_t.start()

    third_t = threading.Thread(target=third_thread, args=(manager, 3, verbose_thrid_thread, debug))
    third_t.daemon = True
    third_t.start()

    while True:
        time.sleep(100)
    

if __name__ == '__main__':
    main()