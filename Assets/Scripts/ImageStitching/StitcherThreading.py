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

        self.active_stitcher = self.stitchers["UDIS"]
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
        self.shared_drone_ids = None
        self.shared_headings = None
        self.known_order = None  # Store the known order of images

        self.processedImageWidth = None
        self.processedImageHeight = None
        self.batchImageWidth = None
        self.batchImageHeight = None

        self.panoram_queue = queue.Queue(1)

        # Set the stitcher to setup the device properly
        self.set_stitcher(self.active_stitcher_type, onlyIHN=False)

    def set_stitcher(self, stitcher_type, onlyIHN):
        """
        Safely switch the active stitcher.
        Waits for both threads to finish their current work before switching.
        """
        
        # Remove devices from GPU
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
        - output (dict): A dictionary containing stitching settings and hyperparameters
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

    def process_stitching(self, images, num_pano_img=3):
        """
        Simplified stitching process using known order from drone IDs.
        No need for homography computation - just stitch based on known order.
        """
        with self.switching_lock2:
            if self.known_order is None or len(self.known_order) != len(images):
                print(f"[WARNING] Known order not set or length mismatch. Expected {len(images)} images.")
                return
            
            # Create order array based on known drone order
            order = np.array(self.known_order)
            
            # For stitching methods that need different approaches
            if self.active_stitcher_type == "CLASSIC":
                # Use simplified compose method that doesn't need homographies
                pano = self.stitch_with_known_order(images, order, num_pano_img)
            elif self.active_stitcher_type == "UDIS":
                # UDIS uses direct image warping
                subset1, subset2 = self.get_subsets_from_order(order, num_pano_img, len(images))
                pano = self.active_stitcher.UDIS_pano(images, subset1, subset2)
            elif self.active_stitcher_type == "NIS":
                subset1, subset2 = self.get_subsets_from_order(order, num_pano_img, len(images))
                # NIS stitch implementation would go here
                pano = None  # Placeholder
            elif self.active_stitcher_type == "REWARP":
                subset1, subset2 = self.get_subsets_from_order(order, num_pano_img, len(images))
                # REWARP stitch implementation would go here
                pano = None  # Placeholder
            
            if pano is not None and self.panoram_queue.empty():
                self.panoram_queue.put(pano)

    def get_subsets_from_order(self, order, num_pano_img, num_images):
        """
        Get image subsets based on head angle and known order.
        Simplified version that uses the headAngle to determine which images to use.
        """
        # Use the headAngle to determine reference image
        angle_per_image = 360.0 / num_images
        ref_idx = int((self.headAngle / angle_per_image) + 0.5) % num_images
        
        odd = num_pano_img % 2

        # print("HERE1", ref_idx, order, num_pano_img, num_images, self.headAngle)
        
        if odd:
            offset = num_pano_img // 2
            subset1 = order[ref_idx-offset:ref_idx+1][::-1] if ref_idx >= offset else np.concatenate([order[ref_idx-offset:], order[:ref_idx+1]])[::-1]
            subset2 = order[ref_idx:ref_idx+offset+1] if ref_idx + offset + 1 <= num_images else np.concatenate([order[ref_idx:], order[:ref_idx+offset+1-num_images]])
        else:
            right_offset = int((self.headAngle % angle_per_image) >= angle_per_image/2)
            offset = num_pano_img // 2
            subset1 = order[ref_idx-offset+1:ref_idx+1][::-1] if ref_idx >= offset-1 else np.concatenate([order[ref_idx-offset+1:], order[:ref_idx+1]])[::-1]
            subset2 = order[ref_idx:ref_idx+offset+right_offset] if ref_idx + offset + right_offset <= num_images else np.concatenate([order[ref_idx:], order[:ref_idx+offset+right_offset-num_images]])
        
        return subset1, subset2

    def stitch_with_known_order(self, images, order, num_pano_img):
        """
        Simplified stitching for when order is known.
        Just arranges images side by side without complex homography computation.
        """
        subset1, subset2 = self.get_subsets_from_order(order, num_pano_img, len(images))
        
        # Simple horizontal concatenation
        selected_indices = np.concatenate([subset1[::-1], subset2[1:]])  # Avoid duplicate ref image
        selected_images = [images[i] for i in selected_indices]
        
        if len(selected_images) == 0:
            return images[0]
        
        # Resize all images to same height
        target_height = self.processedImageHeight if self.processedImageHeight else images[0].shape[0]
        resized_images = []
        for img in selected_images:
            h, w = img.shape[:2]
            new_width = int(w * target_height / h)
            resized_images.append(cv2.resize(img, (new_width, target_height)))
        
        # Concatenate horizontally
        pano = np.hstack(resized_images)
        
        # Resize to target width if needed
        if self.processedImageWidth and pano.shape[1] != self.processedImageWidth:
            pano = cv2.resize(pano, (self.processedImageWidth, target_height))
        
        return pano

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

def first_thread(manager: StitcherManager, num_images=3, debug=False, enable_debug_logging=False):
    """
    This method reads images from the block-based shared memory structure.
    Each block contains: flag (4 bytes), droneId (4 bytes), heading (4 bytes), image data
    """
    
    # Read metadata first to get image dimensions
    metadataSize = 20 + 64 + 1 + 64 + 1 + 4*4 + 1
    metadataMMF = mmap.mmap(-1, metadataSize, "MetadataSharedMemory")
    
    output = readMetadataMemory(metadataMMF)
    batchImageWidth, batchImageHeight, imageCount, manager.processedImageWidth, manager.processedImageHeight = output["Sizes"]

    # ----------------- TODO: Remove hardcoding -----------------
    batchImageWidth = 960
    batchImageHeight = 540
    
    
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
    processedMMF = mmap.mmap(-1, totalProcessedSize, "BlockSharedMemory")
    
    first_loop = True
    
    while True:
        # Update metadata
        output = readMetadataMemory(metadataMMF)
        batchImageWidth, batchImageHeight, imageCount, manager.processedImageWidth, manager.processedImageHeight = output["Sizes"]
        manager.checkHyperparaChanges(output)

        # ----------------- TODO: Remove hardcoding -----------------
        batchImageWidth = 960
        batchImageHeight = 540
        imageCount = num_images
        manager.processedImageWidth = 1920
        manager.processedImageHeight = 1080
        # print(batchImageWidth, batchImageHeight, imageCount, manager.processedImageWidth, manager.processedImageHeight)
        
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
        
        if len(images) > 0:
            # Sort images by drone ID to get known order
            sorted_indices = np.argsort(drone_ids)
            sorted_images = [images[i] for i in sorted_indices]
            sorted_drone_ids = [drone_ids[i] for i in sorted_indices]
            sorted_headings = [headings[i] for i in sorted_indices]
            
            # Store the images and metadata
            with manager.info_lock:
                manager.shared_images = sorted_images
                manager.shared_drone_ids = sorted_drone_ids
                manager.shared_headings = sorted_headings
                # Create known order based on sorted drone IDs
                manager.known_order = [1,2,0]#list(range(len(sorted_images)))
                # Use the first heading as the overall head angle (or compute average)
                manager.headAngle = sorted_headings[0] if len(sorted_headings) > 0 else 0
            
            if enable_debug_logging:
                print(f"[first_thread] Read {len(images)} images, sorted drone IDs: {sorted_drone_ids}, headings: {sorted_headings}")
        
        # Write panorama if available
        if not manager.panoram_queue.empty():
            panorama = manager.panoram_queue.get()
            H, W, _ = panorama.shape
            print(f"Panorama size: {W}x{H}")
            if H != manager.processedImageHeight or W != manager.processedImageWidth:
                try:
                    panorama = cv2.resize(panorama, (manager.processedImageWidth, manager.processedImageHeight))
                    # Save the panorama to disk for verification
                    cv2.imwrite("stitched_panorama.jpg", panorama)
                except:
                    continue
            
            try:
                panoramaMMF = mmap.mmap(-1, manager.processedImageWidth * manager.processedImageHeight * 3 + 4 + 4, "PanoramaSharedMemory")
                
                # Flip the panorama because unity texture starts bottom left
                panorama = cv2.flip(panorama, 0)

                write_memory(panoramaMMF, 0, 4, manager.processedImageWidth * manager.processedImageHeight * 3, panorama)
                del panorama
            except Exception as e:
                if enable_debug_logging:
                    print(f"[first_thread] Error writing panorama to memory: {e}")
                continue
        
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

    if enable_debug:
        print(f"Reading {num_blocks} blocks from mmmf... blockSize={blockSize}, imageSize={imageSize}, imageWidth={imageWidth}, imageHeight={imageHeight}")
    
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

def stitching_thread(manager: StitcherManager, num_pano_img=3, verbose=False, debug=False):
    """
    Simplified stitching thread that uses known order from drone IDs.
    """
    while True:
        if manager.shared_images is None or manager.known_order is None:
            if verbose:
                print("[stitching_thread] Waiting for images and known order...")
            time.sleep(0.4)
            continue
        
        with manager.info_lock:
            images = manager.shared_images
        
        t = time.time()
        
        try:
            manager.process_stitching(images, num_pano_img=num_pano_img)
        except Exception as e:
            print(f"[stitching_thread] Error during stitching: {e}")
        
        if verbose:
            print(f"[stitching_thread] Loop time: {time.time()-t:.3f}s")
        
        if debug:
            break
    
    print("[stitching_thread] Quitting stitching thread")

def readMetadataMemory(metadataMMF :mmap )->dict:
    """
    Reads metadata from a memory-mapped file and returns it as a dictionary.
    """
    
    # Read the integers for image sizes
    metadataMMF.seek(0)
    int_values = struct.unpack('iiiii', metadataMMF.read(20))  # 5 integers

    # Read the string for Stitcher Type
    raw_string = metadataMMF.read(64)
    metadata_string = raw_string.decode('utf-8').rstrip('\x00')

    # Read the boolean isCylindrical
    raw_bool = metadataMMF.read(1)
    metadata_bool = bool(struct.unpack('B', raw_bool)[0])

    # Read the string for BF or FLANN
    raw_string = metadataMMF.read(64)
    matcherType = raw_string.decode('utf-8').rstrip('\x00')

    # Read the boolean for RANSAC or not
    raw_bool = metadataMMF.read(1)
    isRANSAC = bool(struct.unpack('B', raw_bool)[0])

    # Read the integer for check sizes
    checks = struct.unpack('i', metadataMMF.read(4))[0]

    # Read the float for ratio and score
    floats = struct.unpack('ff', metadataMMF.read(8))

    # Read the integer for focal
    focal = struct.unpack('i', metadataMMF.read(4))[0]

    # Read the boolean for onlyIHN
    raw_bool = metadataMMF.read(1)
    onlyIHN = bool(struct.unpack('B', raw_bool)[0])

    return {
        "Sizes": int_values,
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
    Activates the threads and initializes the StitcherManager.
    """

    manager = StitcherManager("cuda")
    verbose_stitching_thread = True
    debug = False
    enable_debug_logging = False  # Set to True for debugging

    # Print the keys of available stitchers
    print("Available stitchers:")
    for key in manager.stitchers.keys():
        print(f"- {key}")

    # Number of image blocks to read (should match numImages in ImageSharing.cs)
    num_images = 3  # Update this to match your configuration
    num_pano_img = 3  # Number of images in the panorama

    first_t = threading.Thread(target=first_thread, args=(manager, num_images, debug, enable_debug_logging))
    first_t.daemon = True
    first_t.start()

    stitch_t = threading.Thread(target=stitching_thread, args=(manager, num_pano_img, verbose_stitching_thread, debug))
    stitch_t.daemon = True
    stitch_t.start()

    while True:
        time.sleep(100)
    

if __name__ == '__main__':
    main()