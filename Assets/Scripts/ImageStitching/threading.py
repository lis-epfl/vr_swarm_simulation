# # import custom_stitching
# import numpy as np
# import cv2
# import glob
# import os
# import torch
# import time
# from transformers import SuperPointForKeypointDetection
# # from torch.quantization import quantize_dynamic
# from numba import jit
# import queue
# import threading
# import mmap
# import struct
# import networkx as nx
# import random


# lock = threading.Lock()
# global stitcher

# global processedImageWidth
# global processedImageHeight
# global panoram_queue
# global headAngle
# # global typeOfStitcher, isCylindrical

# stitcher =0

# class StitcherManager:
#     def __init__(self):
#         self.active_stitcher = None
        
#         self.switching_lock1 = threading.Lock()
#         self.switching_lock2 = threading.Lock()
#         self.info_lock = threading.Lock()

#         self.stitcherTypes = ["CLASSIC", "UDIS", "NIS"]
#         self.cylidnricalWarp = False
#         self.panoram_queue = queue.Queue(1)
#         self.headAngle = 0
#         self.shared_images = None
#         self.shared_images_bool = None

#     def set_stitcher(self, stitcher):
#         """
#         Safely switch the active stitcher.
#         Waits for both threads to finish their current work before switching.
#         """
#         if stitcher in self.stitcherTypes:
#             # Acquire both locks to ensure both threads are idle
#             with self.switching_lock1, self.switching_lock2:
#                 self.active_stitcher = stitcher
#                 print(f"Switched to {stitcher.__class__.__name__}")

#     def process_thread2(self):
#         """
#         Thread 2 operation. Uses lock_thread2 for thread-safe access.
#         """
#         with self.switching_lock1:
#             if self.active_stitcher:
#                 self.active_stitcher.process_images()

#     def process_thread3(self):
#         """
#         Thread 3 operation. Uses lock_thread3 for thread-safe access.
#         """
#         with self.switching_lock2:
#             if self.active_stitcher:
#                 self.active_stitcher.stitch_panorama()

#     def changeCylindrical(self):
#         pass



# def first_thread(manager: StitcherManager, debug = False):
#     """""
#     This method read the images coming from the software. They have three shared memory files to store the images, the panorama and the metadatas.
#     """""
#     global processedImageWidth
#     global processedImageHeight

#     flagPosition = 0
#     processedDataPosition = 4
#     metadataSize = 20 + 64 + 1 # 20 bytes for ints (5x4 bytes) + 64 bytes for string + 1 byte bool
#     metadataMMF = mmap.mmap(-1, metadataSize, "MetadataSharedMemory")

#     # Read first time metadata to initialize the memories:
#     output = readMetadataMemory(metadataMMF)
#     typeOfStitcher, isCylindrical= output["string"], output["boolean"]
    
#     batchImageWidth, batchImageHeight, imageCount, processedImageWidth, processedImageHeight= output["int_values"]
    
#     imageSize, headANglePosition, batchDataPosition, processedImageSize = UpdateValues(batchImageWidth, batchImageHeight, imageCount, processedImageWidth, processedImageHeight)
#     batchMMF = mmap.mmap(-1, batchDataPosition +  imageCount* imageSize, "BatchSharedMemory")
#     processedMMF = mmap.mmap(-1, processedDataPosition + processedImageSize, "ProcessedImageSharedMemory")

#     first_loop = True
#     while True:

#         ### New part
#         output = readMetadataMemory(metadataMMF)
#         typeOfStitcher, isCylindrical= output["string"], output["boolean"]
#         batchImageWidth, batchImageHeight, imageCount, processedImageWidth, processedImageHeight= output["int_values"]
#         imageSize, headANglePosition, batchDataPosition, processedImageSize = UpdateValues(batchImageWidth, batchImageHeight, imageCount, processedImageWidth, processedImageHeight)
        
#         if manager.active_stitcher != typeOfStitcher:
#             manager.set_stitcher(typeOfStitcher)

#         if manager.cylidnricalWarp != isCylindrical:
#             manager.changeCylindrical()

#         try:
#             batchMMF = mmap.mmap(-1, batchDataPosition +  imageCount* imageSize, "BatchSharedMemory")
#             images, images_bool, headAngle = readMemory(batchMMF, flagPosition, headANglePosition, imageCount, batchDataPosition, imageSize, batchImageWidth, batchImageHeight)
#         except:
#             print("problem opening/reading batched memory")
#             continue
        
#         # droneImInd = np.arange(0, images_bool.shape[0])[images_bool]
#         # print(f"Index of the drone images to stitch: {droneImInd}")
#         with manager.info_lock:
#             manager.shared_images = images
#             manager.shared_images_bool = images_bool
#             manager.headAngle = headAngle

#         if not manager.panoram_queue.empty():
#             panorama = manager.panoram_queue.get()
#             H, W, _ = panorama.shape
#             if H != processedImageHeight or W != processedImageWidth:
#                 try:
#                     panorama = cv2.resize(panorama, (processedImageWidth, processedImageHeight))
#                 except:
#                     continue
#             # if debug:
#             #     panoram_queue.put(panorama)
#             #     break
#             try:
#                 processedMMF = mmap.mmap(-1, processedDataPosition + processedImageSize, "ProcessedImageSharedMemory")
#                 write_memory(processedMMF, flagPosition, processedDataPosition, processedImageSize, cv2.flip(panorama, 0))
#                 del panorama
#             except:
#                 print("problem opening/reading processed memory")
#                 continue


#         time.sleep(0.05)

#         if first_loop:
#             first_loop = False
#             time.sleep(1.)

#         if debug:
#             break

# def second_thread(manager: StitcherManager, front_image_index=0, debug= False):
#     """""
#     This method uses some of the above methods to extract the order and the homographies of the paired images.
#     Input:
#         - images: list of NDArrays.
#         - front_image_index: the index of the front image of the pilot
#     """""
#     Hs = None
#     # global shared_images
#     while True:
#         if manager.shared_images is None:
#                 print("Second thread sleep")
#                 time.sleep(0.4)
#                 continue
        
#         with lock:
#             images = manager.shared_images

#         t = time.time()
#         if manager.cylidnricalWarp:
#             outputs, ratios, images = SP_inference_fast(images)
#         else:
#             outputs, ratios = SP_inference_fast(images)

        
#         keypoints, descriptors = keep_best_keypoints(outputs, ratios)
#         t1 = time.time()
#         # keypoints, descriptors = ORB_extraction(images)
#         matches_info, confidences = compute_matches_and_confidences(descriptors, keypoints)
#         t2 = time.time()
#         best_pairs = find_top_pairs(confidences)
#         partial_order = find_cycle_for_360_panorama(confidences, front_image_index, False)[:-1]
#         H, order, inverted = compute_homographies_and_order(keypoints, matches_info, partial_order, Hs)
#         if Hs is not None:
#             for i in range(H.shape[0]):
#                 Hs[i] = smooth_homography(H[i], Hs[i], alpha=0.08)
#         else:
#             Hs = H

#         t3 = time.time()
#         # Ms, order, inverted = compute_affines_and_order(keypoints, matches_info, partial_order)

#         # print("time to extract keypoints:", t1-t)
#         # print("time to compute matches:", t2-t1)
#         # print("time to compute homographies:", t3-t2)

#         # put everything in the queues
#         order_queue.put(order)
#         homography_queue.put(Hs)
#         # homography_queue.put(Ms)
#         direction_queue.put(inverted)
#         print(order)
#         # print(f"Time to compute the Parameters: {time.time()-t}")
#         # time.sleep(0.5)
#         # print("New parameters given")
        
#         if debug:
#             return keypoints, Hs, order, inverted, best_pairs, matches_info, confidences, images
#             # return keypoints, Ms, order, inverted, best_pairs, matches_info, confidences, images

# def third_thread(manager: StitcherManager, num_pano_img=3, debug=False):
#     """""
#     This method uses some of the above methods to stitch a part of the given images based on a criterion that could be the orientation
#     of the pilots head and the desired number of images in the panorama.
#     Input:
#         - images: list of NDArrays.
#         - angle : orientation of the pilots head (in degrees [0,360[?)
#         - num_pano_img : desired number of images in the panorama
#     """""
#     # Try taking the homography and order. Until the queues are empty, keep the homograpies and orders in local variable
#     # Take the order and the ref to compute the panorama

#     global processedImageWidth
#     global processedImageHeight

#     order, Hs, inverted = None, None, None

#     while True:
#         if not manager.homography_queue.empty():
#             del order, Hs, inverted
#             order = manager.order_queue.get()
#             Hs = manager.homography_queue.get()
#             inverted = manager.direction_queue.get()
#             order = np.hstack((order, order, order))
#             Hs = np.concatenate((Hs, Hs, Hs))
#         elif order is None:
#             time.sleep(0.4)
#             continue
        
#         with lock:
#             images = manager.shared_images

#         t = time.time()
#         if manager.cylidnricalWarp:
#             images = [cylindricalWarp(img) for img in images]
        
        
#         subset1, subset2, Hs1, Hs2 = chooseSubsetsAndTransforms(Hs, num_pano_img, order, headAngle)
#         # pano = compose_with_ref(images, Hs1, Hs2, subset1, subset2, inverted)
#         pano = compose_with_defined_size(images, Hs1, Hs2, subset1, subset2, inverted, panoWidth=processedImageWidth, panoHeight=processedImageHeight)
#         # pano = affineStitching(images, Hs1, Hs2, subset1, subset2, inverted)
        
#         panoram_queue.put(pano)

#         # print(f"Time to make the Panorama: {time.time()-t}")
#         # print("Panorama Given")
#         if debug:
#             if inverted:
#                 print("inverted")
#             print(f"Order of the right : {subset2}")
#             print(f"Order of the left : {subset1}")

#             break

# def readMetadataMemory(metadataMMF :mmap )->dict:
    
#     # Read the integers
#     metadataMMF.seek(0)
#     int_values = struct.unpack('iiiii', metadataMMF.read(20))  # 5 integers

#     # Read the string
#     raw_string = metadataMMF.read(64)
#     metadata_string = raw_string.decode('utf-8').rstrip('\x00')  # Remove padding

#     # Read the boolean
#     raw_bool = metadataMMF.read(1)
#     metadata_bool = bool(struct.unpack('B', raw_bool)[0])  # Unpack as unsigned char

#     # Return all parsed metadata
#     return {
#         "int_values": int_values,  # Tuple of 5 integers
#         "string": metadata_string,
#         "boolean": metadata_bool
#     }

# @jit(nopython=True) 
# def UpdateValues(batchImageWidth, batchImageHeight, imageCount, processedImageWidth, processedImageHeight):
    
#     imageSize = batchImageWidth*batchImageHeight*3
#     headAnglePosition = 5
#     batchDataPosition = headAnglePosition+ imageCount

#     processedImageSize = processedImageWidth*processedImageHeight*3

#     return imageSize, headAnglePosition, batchDataPosition, processedImageSize
 
# def readMemory(batchMMF, batchFlagPosition, headANglePosition, imageCount, batchDataPosition, imageSize, imageWidth, imageHeight):
#     """
#         Read Memory shared with Unity code. If the flag is 0, we can access data.

#         Input:
#             - batchMMF: mmap object for the shared memory
#             - batchFlagPosition: position of the flag in the memory. In our case: 0
#             - imageCount: number of images
#             - batchDataPosition: position of the first image in memory. In our case: 1 + imageCount (1 byte for the flag + imageCount bytes for the boolean)
#             - imageSize: The full image size (imageHeight * imageWidth * 3)
#             - imageWidth
#             - imageHeight
#     """
    
#     images = []
#     while True:
#         # Read the flag to check if Unity has written new images
#         batchMMF.seek(batchFlagPosition)
#         flag = struct.unpack('B', batchMMF.read(1))[0]

#         if flag == 0:  # Unity isn't writing new images
            
#             # Flag to 1, indicating we are reading
#             batchMMF.seek(batchFlagPosition)
#             batchMMF.write(struct.pack('B', 1))

#             batchMMF.seek(1)
#             boolean_list = [bool(b) for b in batchMMF.read(imageCount)]
#             boolean_array=np.array(boolean_list)

#             batchMMF.seek(headANglePosition)
#             headAngle = struct.unpack('f', batchMMF.read(4))[0]

#             for i in range(imageCount):
#                 if not boolean_array[i]:
#                     continue
#                 # Read each image sequentially from the shared memory
#                 batchMMF.seek(batchDataPosition + i * imageSize)
#                 image_data = batchMMF.read(imageSize)

#                 # Convert the byte array into a numpy array
#                 image = np.frombuffer(image_data, dtype=np.uint8)
#                 image = image.reshape((imageHeight, imageWidth, 3))  # Reshape to RGB format
#                 images.append(cv2.flip(image, 0))

#             # Reset flag to 0, indicating we've read the images
#             batchMMF.seek(batchFlagPosition)
#             batchMMF.write(struct.pack('B', 0))

#             return images, boolean_array, headAngle

# def write_memory(processedMMF, processedFlagPosition, processedDataPosition, processedImageSize, image_data):
#     """
#     Write an image to shared memory with Unity.

#     Inputs:
#         - processedMMF: mmap object for the shared memory.
#         - processedFlagPosition: position of the flag in the memory
#         - processedDataPosition: position to start writing the image data.
#         - processedImageSize: expected size of the image data.
#         - image_data: numpy array of the image to write.
#     """
#     while True:
#         # Read the flag to check if Unity is ready for new data
#         processedMMF.seek(processedFlagPosition)
#         flag = struct.unpack('i', processedMMF.read(4))[0]

#         if flag == 0:  # Unity isn't writing new images
#             # Set flag to 1, indicating we're writing
#             processedMMF.seek(processedFlagPosition)
#             processedMMF.write(struct.pack('i', 1))

#             # Convert image to byte array and check size
#             image_bytes = image_data.tobytes()
#             if len(image_bytes) != processedImageSize:
#                 raise ValueError(f"Image size mismatch: expected {processedImageSize}, got {len(image_bytes)}")

#             # Write the image bytes to shared memory
#             processedMMF.seek(processedDataPosition)
#             processedMMF.write(image_bytes)

#             # Reset flag to 0, indicating we've written the image
#             processedMMF.seek(processedFlagPosition)
#             processedMMF.write(struct.pack('i', 0))
#             break

# def main():

#     imageWidth = 300
#     imageHeight = 300

#     f = 160

#     cam_mat = np.array([[f, 0, imageWidth/2], 
#                         [0, f, imageHeight/2],
#                         [0, 0, 1]])

#     stitcher = custom_stitcher_SP(camera_matrix=cam_mat, warp_type="cylindrica", algorithm=1, trees=5, checks=50, ratio_thresh=0.7, score_threshold=0.05, device="cuda")

#     first_thread = threading.Thread(target=stitcher.first_thread, args=(False,))
#     first_thread.daemon = True
#     first_thread.start()

#     sec_thread = threading.Thread(target=stitcher.second_thread, args=(0, False))
#     sec_thread.daemon = True
#     sec_thread.start()

#     third_thread = threading.Thread(target=stitcher.third_thread, args=(0, 3, False))
#     third_thread.daemon = True
#     third_thread.start()

#     while True:
#         time.sleep(100)
    

# if __name__ == '__main__':
    
#     # main function, the final result
#     main()

#     # Test reading the images from shared memory and write an image in the panorama memory
#     # test_reading_writing()

#     # Test stitcher with own images
#     # test_stitcher()

#     # Test threading function
#     # test_threading()

#     # Test one stitched camera
#     # front_image_index = 0
#     # angle = 0
#     # num_pano_img = 3

#     # test_one_image(front_image_index, angle, num_pano_img)