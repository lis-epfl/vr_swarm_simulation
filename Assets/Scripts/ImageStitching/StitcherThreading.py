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

# Activate environnement
# cmd
# cd Assets\Scripts\ImageStitching 
# python StitcherThreading.py


class StitcherManager:
    def __init__(self, device ="cpu"):

        self.stitchers = {
            "CLASSIC": BaseStitcher(algorithm=1, trees=5, checks=50, ratio_thresh=0.7, score_threshold=0.15, device=device),
            "UDIS": UDISStitcher(),
            "NIS": NISStitcher(),  # Replace with your NISStitcher instance if implemented
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

        self.stitcherTypes = ["CLASSIC", "UDIS", "NIS"]
        self.cylidnricalWarp = False
        self.isRANSAC = False
        self.headAngle = 0
        self.shared_images = None
        self.shared_images_bool = None

        self.processedImageWidth = None
        self.processedImageHeight = None

        self.order_queue = queue.Queue(1)
        self.homography_queue = queue.Queue(1)
        self.direction_queue = queue.Queue(1)
        self.panoram_queue = queue.Queue(1)

    def set_stitcher(self, stitcher_type):
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

    def checkHyperparaChanges(self, output : dict):
        typeOfStitcher, isCylindrical, matcherType, isRANSAC = output["typeOfStitcher"], output["isCylindrical"], output["matcherType"], output["isRANSAC"]
        
        if (self.active_stitcher_type != typeOfStitcher and typeOfStitcher in self.stitcherTypes) or self.active_stitcher.cylindricalWarp != isCylindrical:
            with self.switching_lock1:
                with self.switching_lock2:
                    self.set_stitcher(typeOfStitcher)
                    self.changeCylindrical(isCylindrical)
                    self.changeCalculationsHyperpara(output)
        
        elif self.active_stitcher.active_matcher_type != matcherType or self.active_stitcher.isRANSAC != isRANSAC:
            with self.switching_lock1:
                self.changeCalculationsHyperpara(output)

    def process_thread2(self, images, front_image_index,Hs, verbose, debug):
        """
        Thread 2 operation. Uses lock_thread2 for thread-safe access.
        """
        with self.switching_lock1:
            # if self.active_stitcher is not None:
            Hs, order, inverted = self.active_stitcher.findHomographyOrder(images, front_image_index, Hs, verbose, debug)
            self.order_queue.put(order)
            self.homography_queue.put(Hs)
            self.direction_queue.put(inverted)
            print(order)
            return Hs


    def process_thread3(self, images, order, Hs, inverted, num_pano_img=3, verbose= False):
        """
        Thread 3 operation. Uses lock_thread3 for thread-safe access.
        """
        with self.switching_lock2:
            # if self.active_stitcher:

            if self.active_stitcher_type == "CLASSIC":
                pano = self.active_stitcher.stitch(images, order, Hs, inverted, self.headAngle, self.processedImageWidth, self.processedImageHeight, num_pano_img=num_pano_img, verbose=verbose)
            elif self.active_stitcher_type == "UDIS":
                pano = self.active_stitcher.stitch(images, order, Hs, inverted ,self.headAngle, num_pano_img=num_pano_img, verbose=verbose)
            elif self.active_stitcher_type == "NIS":
                # pano = self.active_stitcher.stitch(images, self.headAngle, order, num_pano_img=num_pano_img, verbose=verbose)
                pano = self.active_stitcher.stitch(images, order, Hs, inverted , self.headAngle, num_pano_img=num_pano_img, verbose=verbose)
                # pano = np.zeros((self.processedImageHeight, self.processedImageWidth, 3))
            if self.panoram_queue.empty():
                self.panoram_queue.put(pano)

    def changeCylindrical(self, isCylindrical):
        
        self.active_stitcher.cylindricalWarp = isCylindrical
        self.active_stitcher.points_remap = None
        pass

    def changeCalculationsHyperpara(self, output):
        self.active_stitcher.active_matcher_type = output["matcherType"]
        self.active_stitcher.isRANSAC = output["isRANSAC"]
        print( output["isRANSAC"], output["matcherType"])
        pass

def first_thread(manager: StitcherManager, debug = False):
    """""
    This method read the images coming from the software. They have three shared memory files to store the images, the panorama and the metadatas.
    """""
    flagPosition = 0
    processedDataPosition = 4
    metadataSize = 20 + 64 + 1+ 64+ 1 # 20 bytes for ints (5x4 bytes) + 64 bytes for string + 1 byte bool
    metadataMMF = mmap.mmap(-1, metadataSize, "MetadataSharedMemory")

    # Read first time metadata to initialize the memories:
    output = readMetadataMemory(metadataMMF)
    typeOfStitcher, isCylindrical = output["typeOfStitcher"], output["isCylindrical"]
    
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

    # Return all parsed metadata
    return {
        "Sizes": int_values,  # Tuple of 5 integers
        "typeOfStitcher": metadata_string,
        "isCylindrical": metadata_bool,
        "matcherType" : matcherType,
        "isRANSAC" :isRANSAC
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

    imageWidth = 300
    imageHeight = 300

    f = 160

    cam_mat = np.array([[f, 0, imageWidth/2], 
                        [0, f, imageHeight/2],
                        [0, 0, 1]])
    
    manager = StitcherManager("cuda")
    verbose = False
    debug = False

    first_t = threading.Thread(target=first_thread, args=(manager, debug))
    first_t.daemon = True
    first_t.start()

    sec_t = threading.Thread(target=second_thread, args=(manager, 0, verbose, debug))
    sec_t.daemon = True
    sec_t.start()

    third_t = threading.Thread(target=third_thread, args=(manager, 3, verbose, debug))
    third_t.daemon = True
    third_t.start()

    while True:
        time.sleep(100)
    

if __name__ == '__main__':
    
    # main function, the final result
    main()

    # Test reading the images from shared memory and write an image in the panorama memory
    # test_reading_writing()

    # Test stitcher with own images
    # test_stitcher()

    # Test threading function
    # test_threading()

    # Test one stitched camera
    # front_image_index = 0
    # angle = 0
    # num_pano_img = 3

    # test_one_image(front_image_index, angle, num_pano_img)