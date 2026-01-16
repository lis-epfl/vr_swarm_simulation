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

# --- UDIS ---
HAS_UDIS = False
try:
    sys.path.append(os.path.abspath("UDIS2_main\Warp\Codes"))
    import UDIS2_main.Warp.Codes.utils_udis as udis_utils
    import UDIS2_main.Warp.Codes.utils_udis.torch_DLT as torch_DLT
    import UDIS2_main.Warp.Codes.grid_res as grid_res
    from UDIS2_main.Warp.Codes.network import build_output_model, get_stitched_result, Network, build_new_ft_model
    from UDIS2_main.Warp.Codes.loss import cal_lp_loss2
    from UDISStitcher import *
    HAS_UDIS = True
except ImportError as e:
    print("UDIS modules could not be imported. UDIS stitcher will not be available.")
    print(e)

# --- NIS ---
HAS_NIS = False
try:
    sys.path.append(os.path.abspath("Neural_Image_Stitching_main"))
    import Neural_Image_Stitching_main.srwarp
    import Neural_Image_Stitching_main.utils as nis_utils
    from Neural_Image_Stitching_main.models.ihn import *
    from Neural_Image_Stitching_main.models import *
    from Neural_Image_Stitching_main import stitch
    import Neural_Image_Stitching_main.pretrained
    from NISStitcher import *
    HAS_NIS = True
except ImportError as e:
    print("NIS modules could not be imported. NIS stitcher will not be available.")
    print(e)

# --- REWARP ---
HAS_REWARP = False
try:
    sys.path.append(os.path.abspath("Residual_Elastic_Warp_main"))
    import Residual_Elastic_Warp_main.models
    import Residual_Elastic_Warp_main.utils
    from REStitcher import *
    HAS_REWARP = True
except ImportError as e:
    print("REWARP modules could not be imported. REWARP stitcher will not be available.")
    print(e)

# Activate environnement
# cmd
# cd Assets\Scripts\ImageStitching
# python StitcherThreading.py

class StitcherManager:
    def __init__(self, device ="cpu"):

        self.stitchers = {
            "CLASSIC": BaseStitcher(algorithm=1, trees=5, checks=50, ratio_thresh=0.7, score_threshold=0.05, device=device),
            "UDIS": UDISStitcher() if HAS_UDIS else None,
            "NIS": NISStitcher() if HAS_NIS else None,
            "REWARP": REStitcher() if HAS_REWARP else None,
        }
        

        # Manually remove models of NIS from GPU because they load them directly on GPU
        if HAS_NIS:
            self.stitchers["NIS"].model.cpu(), self.stitchers["NIS"].H_model.cpu()

        self.active_stitcher = self.stitchers["CLASSIC"]
        self.active_stitcher_type = "UDIS"
        self.active_matcher_type = "BF"
        self.device = device
        
        self.switching_lock1 = threading.Lock()
        self.switching_lock2 = threading.Lock()
        self.info_lock = threading.Lock()

        self.stitcherTypes = list(self.stitchers.keys())
        self.cylidnricalWarp = False
        self.isRANSAC = False   
        self.headAngle = 0
        self.shared_images = None
        self.shared_images_bool = None
        self.shared_drone_ids = None
        self.shared_headings = None

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
            except Exception as e:
                if verbose:
                    print(f"Error in process_thread2: {e}")
                return None
            self.order_queue.put(order)
            self.homography_queue.put(Hs)
            self.direction_queue.put(inverted)
            return Hs

    def process_thread3(self, images, order, Hs, inverted, num_pano_img=3):
        """
        Thread 3 operation. Uses lock_thread3 for thread-safe access. Stitch the images by using information from the queues
        """
        # print active stitcher
        print(f"[third_thread] Using stitcher: {self.active_stitcher_type}")

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

def first_thread(manager: StitcherManager, num_images=1, debug=False, enable_debug_logging=False):
    """
    This method reads images from the block-based shared memory structure.
    Each block contains: flag (4 bytes), droneId (4 bytes), heading (4 bytes), image data
    """
    
    # Read metadata first to get image dimensions
    metadataSize = 20 + 64 + 1 + 64 + 1 + 4*4 + 1
    metadataMMF = mmap.mmap(-1, metadataSize, "MetadataSharedMemory")
    
    output = readMetadataMemory(metadataMMF)
    batchImageWidth, batchImageHeight, imageCount, manager.processedImageWidth, manager.processedImageHeight = output["Sizes"]
    
    # Calculate block-based memory layout
    metadataSize_per_block = 12  # flag (4) + droneId (4) + heading (4)
    imageSize = batchImageWidth * batchImageHeight * 3  # RGB24
    blockSize = metadataSize_per_block + imageSize
    totalProcessedSize = num_images * blockSize
    
    if enable_debug_logging:
        print(f"[first_thread] Initializing with {num_images} image blocks")
        print(f"[first_thread] Block size: {blockSize} bytes (metadata: {metadataSize_per_block}, image: {imageSize})")
        print(f"[first_thread] Total memory size: {totalProcessedSize} bytes")
    
    # Open the block-based shared memory (same one ImageSharing.cs uses)
    processedMMF = mmap.mmap(-1, totalProcessedSize, "ProcessedImageSharedMemory")
    
    first_loop = True
    
    while True:
        # Update metadata
        output = readMetadataMemory(metadataMMF)
        batchImageWidth, batchImageHeight, imageCount, manager.processedImageWidth, manager.processedImageHeight = output["Sizes"]
        manager.checkHyperparaChanges(output)
        
        # Read images from block-based memory
        try:
            images, drone_ids, headings = read_block_memory(
                processedMMF, 
                num_images, 
                blockSize, 
                metadataSize_per_block,
                imageSize, 
                batchImageWidth, 
                batchImageHeight,
                enable_debug_logging
            )
        except Exception as e:
            if enable_debug_logging:
                print(f"[first_thread] Error reading block memory: {e}")
            time.sleep(0.05)
            continue
        
        # Store the images and metadata
        with manager.info_lock:
            manager.shared_images = images
            manager.shared_drone_ids = drone_ids
            manager.shared_headings = headings
            # Use the first heading as the overall head angle (or compute average)
            if len(headings) > 0:
                manager.headAngle = headings[0]
            # Create a boolean array indicating which images are valid
            manager.shared_images_bool = np.ones(len(images), dtype=bool)
        
        if enable_debug_logging and len(images) > 0:
            print(f"[first_thread] Read {len(images)} images, drone IDs: {drone_ids}, headings: {headings}")
        
        # Write panorama if available
        if not manager.panoram_queue.empty():
            panorama = manager.panoram_queue.get()
            H, W, _ = panorama.shape
            if H != manager.processedImageHeight or W != manager.processedImageWidth:
                try:
                    panorama = cv2.resize(panorama, (manager.processedImageWidth, manager.processedImageHeight))
                except:
                    continue
            
            # Write panorama back to a separate output memory (you can use the old processedMMF structure)
            # For now, we'll skip writing back since ImageSharing.cs is reading individual images
            # If you need to write panorama back, create a separate memory mapped file
            del panorama
        
        time.sleep(0.05)
        
        if first_loop:
            first_loop = False
            time.sleep(1.)
        
        if debug:
            break

def read_block_memory(processedMMF, num_blocks, blockSize, metadataSize, imageSize, imageWidth, imageHeight, enable_debug=False):
    """
    Reads images from block-based shared memory.
    
    Block layout for each image:
        - int flag (4 bytes)
        - int droneId (4 bytes) 
        - float heading (4 bytes)
        - image data (imageSize bytes)
    
    Returns:
        - images: list of numpy arrays
        - drone_ids: list of drone IDs
        - headings: list of heading angles
    """
    images = []
    drone_ids = []
    headings = []
    
    for block_idx in range(num_blocks):
        blockOffset = block_idx * blockSize
        
        # Read flag
        processedMMF.seek(blockOffset)
        flag_bytes = processedMMF.read(4)
        if len(flag_bytes) != 4:
            continue
        flag = struct.unpack('i', flag_bytes)[0]
        
        if enable_debug:
            print(f"[read_block_memory] Block {block_idx}: flag={flag}, offset={blockOffset}")
        
        # Only read if flag is 0 (ready)
        if flag == 0:
            # Set flag to 1 (busy reading)
            processedMMF.seek(blockOffset)
            processedMMF.write(struct.pack('i', 1))
            
            # Read droneId
            processedMMF.seek(blockOffset + 4)
            droneId = struct.unpack('i', processedMMF.read(4))[0]
            
            # Read heading
            processedMMF.seek(blockOffset + 8)
            heading = struct.unpack('f', processedMMF.read(4))[0]
            
            # Read image data
            processedMMF.seek(blockOffset + metadataSize)
            image_data = processedMMF.read(imageSize)
            
            if len(image_data) == imageSize:
                # Convert to numpy array
                image = np.frombuffer(image_data, dtype=np.uint8)
                image = image.reshape((imageHeight, imageWidth, 3))
                
                images.append(image)
                drone_ids.append(droneId)
                headings.append(heading)
                
                if enable_debug:
                    print(f"[read_block_memory] Successfully read block {block_idx}: droneId={droneId}, heading={heading:.2f}")
            
            # Reset flag to 0 (ready for next write)
            processedMMF.seek(blockOffset)
            processedMMF.write(struct.pack('i', 0))
    
    return images, drone_ids, headings

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

def main():
    """
    Activates the three threads and initialize the STitcherManager.
    """

    imageWidth = 1920
    imageHeight = 1080

    f = 160

    cam_mat = np.array([[f, 0, imageWidth/2], 
                        [0, f, imageHeight/2],
                        [0, 0, 1]])
    
    manager = StitcherManager("cuda")
    verbose_second_thread = True
    verbose_thrid_thread = True
    debug = False
    enable_debug_logging = False  # Set to True for debugging

    # Print the keys of available stitchers
    print("Available stitchers:")
    for key in manager.stitchers.keys():
        print(f"- {key}")

    # To test only stitch time when order is known
    for key in manager.stitchers.keys():
        if manager.stitchers[key] is not None:
            manager.stitchers[key].known_order = [0,1,2]
            print(f"Set known_order for {key} stitcher.")

    # Number of image blocks to read (should match numImages in ImageSharing.cs)
    num_images = 1  # Update this to match your configuration

    first_t = threading.Thread(target=first_thread, args=(manager, num_images, debug, enable_debug_logging))
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