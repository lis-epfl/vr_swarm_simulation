# import custom_stitching
import numpy as np
import cv2
import glob
import os
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


lock = threading.Lock()


def plot_graph_with_opencv(cycle, node_positions):
    # Create a white canvas for plotting
    img_size = 600
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    # Define colors for nodes and edges
    node_color = (0, 0, 255)  # Red
    edge_color = (0, 255, 0)  # Green

    # Draw edges for the cycle
    for i in range(len(cycle)):
        node1 = cycle[i]
        node2 = cycle[(i + 1) % len(cycle)]  # Connect to next node, with wraparound to the start
        pos1 = node_positions[node1]
        pos2 = node_positions[node2]
        cv2.line(img, pos1, pos2, edge_color, 2)

    # Draw nodes
    for node in cycle:
        pos = node_positions[node]
        cv2.circle(img, pos, 10, node_color, -1)
        # Add node label (optional)
        cv2.putText(img, str(node), (pos[0] + 10, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Show the image
    cv2.imshow("360-Degree Panorama Graph", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_cycle_for_360_panorama(confidences, start_node, plot = False):
    nb_img = confidences.shape[0]  # Number of images
    G = nx.Graph()
    
    # Add nodes and weighted edges from the confidence matrix (undirected graph)
    G.add_nodes_from(range(nb_img))
    for i in range(nb_img):
        for j in range(i + 1, nb_img):
            G.add_edge(i, j, weight=confidences[i, j])

    # If no start_node is provided, use heuristic to choose the best starting point
    if start_node is None:
        # Heuristic: choose the node with the highest average confidence value
        avg_confidences = np.mean(confidences, axis=1)  # Average confidence per image
        start_node = np.argmax(avg_confidences)  # Choose the image with the highest average confidence
    
    # Greedily construct a cycle that maximizes confidence
    cycle = [start_node]  # Start with the selected node
    visited = [False] * nb_img
    visited[start_node] = True
    
    # Forward construction of the cycle
    while len(cycle) < nb_img:
        current_node = cycle[-1]
        best_next_node = None
        best_confidence = -1

        # Find the next unvisited node with the highest confidence
        for neighbor in range(nb_img):
            if not visited[neighbor] and confidences[current_node, neighbor] > best_confidence:
                best_next_node = neighbor
                best_confidence = confidences[current_node, neighbor]

        # Add the next node to the cycle
        if best_next_node is not None:
            cycle.append(best_next_node)
            visited[best_next_node] = True

    # Close the cycle by connecting the last node to the first node
    cycle.append(cycle[0])

    # Generate random positions for nodes for visualization
    random.seed(42)  # For reproducibility
    node_positions = {i: (random.randint(50, 550), random.randint(50, 550)) for i in range(nb_img)}

    if plot:
        plot_graph_with_opencv(cycle, node_positions)
    
    return cycle



@jit(nopython=True) 
def apply_homographies(Hs, corners):
    """
    Apply the homographies to image corners and return the transformed corners.
    Optimized with Numba for faster execution.
    """

    num_images, num_corners = Hs.shape[0], 4
    tot_points = num_corners*num_images
    all_corners = np.zeros((2, tot_points), dtype=np.float32)  # Store 2D points
    H_accum = np.zeros((num_images, 3, 3), dtype=np.float32)
    
    H = np.eye(3, dtype=np.float32)  # Initial homography matrix
    for i in range(num_images):
        H = np.dot(Hs[i].astype(np.float32), H)  # Update the homography
        new_corners = np.dot(H, corners) # Shape of 3x4
        all_corners[:, i*num_corners: (i+1)*num_corners] = new_corners[:-1]/new_corners[-1]
        H_accum[i]=H
            
    return all_corners, H_accum

@jit(nopython=True) 
def invert_matrices(Hs:np.ndarray)->np.ndarray:
    Hs_inverted = np.zeros_like(Hs)
    num_mat = Hs_inverted.shape[0]
    for i in range(num_mat):
        Hs_inverted[i]=np.linalg.inv(Hs[i])
    return Hs_inverted

class custom_stitcher_SP:
    def __init__(self, camera_matrix, warp_type="cylindrical", full_cylinder = True, algorithm=1, trees=5, checks=50, ratio_thresh = 0.7, score_threshold = 0.2, device = "cpu"):
        
        # SuperPoint model initialization that is used to extract features from images
        self.model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
        # self.model = quantize_dynamic(
        #     self.model,  # the original model
        #     {torch.nn.Conv2d},  # layers to quantize
        #     dtype=torch.qint8  # quantization data type
        #     )
        self.model.eval()
        self.device = torch.device(device)
        
        self.H_SP = 480
        self.W_SP = 640
        self.score_threshold = score_threshold

        self.model.to(self.device)

        # Flann parameters. Algorithm that match the different keypoints between images based 
        # on the descriptors
        self.index_params = dict(algorithm=algorithm, trees=trees)
        self.search_params = dict(checks=checks) 

        # Create a FLANN Matcher
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        self.BF = cv2.BFMatcher()
        self.ratio_thresh = ratio_thresh

        # Camera informations
        self.camera_matrix = camera_matrix

        # Warp informations
        self.warp_type = warp_type
        self.full_cylinder = full_cylinder

        # T keep in memory the "remap matrix"
        self.points_remap = None

        # Thread queues
        self.homography_queue = queue.Queue(1)
        self.order_queue = queue.Queue(1)
        self.direction_queue = queue.Queue(1)
        self.panoram_queue = queue.Queue(1)
        self.shared_images = None
        self.shared_images_bool = None

    def cylindricalWarp(self, img):

        """ 
        taken from: https://github.com/saurabhkemekar/Image-Mosaicing/blob/master/cylinder_stiching.py

        Warps an image in cylindrical coordinate based on the intrinsic camera matrix.
        """
        if self.points_remap is None:
            K = self.camera_matrix
            foc_len = (K[0][0] +K[1][1])/2
            cylinder = np.zeros_like(img)
            temp = np.mgrid[0:img.shape[1],0:img.shape[0]]
            x,y = temp[0],temp[1]
            theta= (x- K[0][2])/foc_len # angle theta
            h = (y-K[1][2])/foc_len # height
            p = np.array([np.sin(theta),h,np.cos(theta)])
            p = p.T
            p = p.reshape(-1,3)
            image_points = K.dot(p.T).T
            points = image_points[:,:-1]/image_points[:,[-1]]
            self.points_remap = points.reshape(img.shape[0],img.shape[1],-1).astype(np.float32)
        cylinder = cv2.remap(img, (self.points_remap[:, :, 0]), (self.points_remap[:, :, 1]), cv2.INTER_LINEAR)
        _, thresh = cv2.threshold(cv2.cvtColor(cylinder, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        
        return cylinder[y:y+h, x:x+w]
    
    def SP_inference_fast(self, images):
        
        """""
        This method uses Superpoint model provided by Hugging Face to extract the keypoints and the descriptors associated. 
        It differs from the above function in the images type and in their processing for faster computation using opencv
        Input: 
            - images : a list of NDArrays images that should be in RGB format
        Outputs:
            - DIctionary : outputs of the model, containing the keypoints, scores and descriptors in numpy arrays
            - tuple : the ratios between the shape of the initial or the cylindrical images and the shape of the data
            taken by Superpoint
            - images : When self.warp_type== "cylindrical" then it outputs the cylindrical images
        
        """""
        if self.warp_type == "cylindrical":
            images_cyl = [self.cylindricalWarp(img) for img in images]
            H, W = images_cyl[0].shape[:2]
            if W<640 or H<480:
                inputs = [cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR) for img in images_cyl]
            else:
                inputs = [cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA) for img in images_cyl]
        else:
            H, W = images[0].shape[:2]
            if W<640 or H<480:
                inputs = [cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR) for img in images]
            else:   
                inputs = [cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA) for img in images]
            
        
        rescale_factor = 0.003921568859368563
        inputs = torch.FloatTensor(np.array(inputs)).permute(0,3,1,2).to("cuda")*rescale_factor
        with torch.no_grad():
            outputs = self.model(inputs)
        
        keypoints, scores, descriptors = outputs['keypoints'].cpu().numpy(), outputs['scores'].cpu().numpy(), outputs['descriptors'].cpu().numpy()

        ratio_y, ratio_x = H/self.H_SP, W/self.W_SP

        if self.warp_type == "cylindrical":
            return {'keypoints': keypoints, 'scores': scores, 'descriptors': descriptors}, (ratio_x, ratio_y), images_cyl

        images = [np.array(image) for image in images]
        return {'keypoints': keypoints, 'scores': scores, 'descriptors': descriptors}, (ratio_x, ratio_y)

    def keep_best_keypoints(self, outputs, ratios):
        # Extract tensors from the outputs dictionary
        kpts, scores, dpts = outputs['keypoints'], outputs['scores'],  outputs['descriptors']

        # Get mask of valid scores
        valid_mask = scores > self.score_threshold

        # Apply mask to keypoints and descriptors
        valid_keypoints = [(kpts[i][valid_mask[i]]*np.array([ratios[0],ratios[1]])).astype(int) for i in range(kpts.shape[0])]

        valid_descriptors = [dpts[i][valid_mask[i]] for i in range(dpts.shape[0])]  # List of valid descriptors for each image
        
        return valid_keypoints, valid_descriptors
    
    def FLANN_matching(self, descriptor1, descriptor2, k=2):
        # descriptors must be numpy arrays
        knn_matches = self.flann.knnMatch(descriptor1, descriptor2, k=k)
        
        # Local variables are faster than looking at the attribute
        ratio_thresh=self.ratio_thresh
        
        # Uses the chosen ratio threshold to keep the best matches based on the Lowe's ratio test
        # It keeps only the matches that have a large enough distance between the two closest neighbours
        return  [m for m, n in knn_matches if m.distance < ratio_thresh * n.distance]
    
    def BF_matching(self, descriptor1, descriptor2, k=2):
        # descriptors must be numpy arrays
        knn_matches = self.BF.knnMatch(descriptor1, descriptor2, k=k)
        
        # Local variables are faster than looking at the attribute
        ratio_thresh=self.ratio_thresh
        
        # Uses the chosen ratio threshold to keep the best matches based on the Lowe's ratio test
        # It keeps only the matches that have a large enough distance between the two closest neighbours
        return  [m for m, n in knn_matches if m.distance < ratio_thresh * n.distance]
    
    def compute_matches_and_confidences(self, descriptors, keypoints):
    
        nb_img = len(descriptors)
        matches_info = []
        
        confidences = np.zeros((nb_img, nb_img))

        for i in range(nb_img):
            desc1= descriptors[i]
            for j in range(i + 1, nb_img):
                
                desc2 = descriptors[j]

                if desc1.size == 0 or desc2.size == 0:
                    continue

                matches = self.FLANN_matching(desc1, desc2)
                # matches = self.BF_matching(desc1, desc2)
                num_matches = len(matches)

                H1, H2 = None, None

                # It was originally made to compute Homographies (needs 4 matches at least to be computed)
                # This should be removed
                if num_matches> 4:
                    # src_p = np.float32([keypoints[i][m.queryIdx] for m in matches]).reshape(-1, 2)
                    # dst_p = np.float32([keypoints[j][m.trainIdx] for m in matches]).reshape(-1, 2)
                    # H1, mask1 = cv2.findHomography(dst_p, src_p, cv2.RANSAC, ransacReprojThreshold=3)
                    # H2, mask2 = cv2.findHomography(src_p, dst_p, cv2.RANSAC, ransacReprojThreshold=3)
                    # max_inliers = max(np.sum(mask1.ravel()), np.sum(mask2.ravel()))
                    # conf = max_inliers / (8 + 0.3 * num_matches)

                    # This is not the exact same equation as the one given by the opencv
                    # function. This could be removed and just plug the num_matches in 
                    # the confidence matrix 
                    conf = num_matches#num_matches / (8 + 0.3 * num_matches)

                    confidences[i, j], confidences[j, i] = conf, conf

                # Store matches information
                # Maybe remove n_matches because useless
                matches_info.append({
                    'image1_index': i,
                    'image2_index': j,
                    'matches': matches,
                    'H1': H1, 
                    'H2': H2
                    # "n_matches": len(matches)
                })
        
        return matches_info, confidences

    def find_top_pairs(self, conf_matrix):
        
        num_images = conf_matrix.shape[0]
        top_pairs = []

        for i in range(num_images):
            # Get the row for the current image and the corresponding confidences
            confidences = conf_matrix[i]

            # Get the indices of the top 2 confidences
            top_2_indices = np.argsort(confidences)[-2:][::-1]

            top_pairs.append(top_2_indices.tolist())
            

        return top_pairs
    
    def find_partial_image_order(self, best_pairs, confidences, ref = 0):
        """""
        This method computes the partial order of the images in the panorama based on the best pairs list. It is the partial order
        because we don't know if the image i is on the right or on the left of image i+1 on the final pano

        Input: 
            - best_pairs : a list of the best pairs of images. More precisely it means that the i-th element contains 
            the two best matched images of image i
            - confidences: confidence matrix used when there are 3 images in the pano. It is uselfull because we take
            the image with the highest mean confidence as the center image.
            - ref: the index of reference image, which is the one considered in front of the pilot when looking forward
        Outputs:
            - order: the partial order of the images. There is two orders possible but we can easily obtained the other by
            doing order[::-1]
        """""
        
        num_img = len(best_pairs)
        order = np.zeros(num_img, dtype= np.uint8)
        order[0] = ref
        idx_set = {ref}
        reference = ref
            
        for idx in range(1,num_img):
            pair = best_pairs[reference][0]
            if pair not in idx_set:
                order[idx]=pair
                reference = pair
                idx_set.add(pair)
            else:
                pair = best_pairs[reference][1]
                order[idx]=pair
                reference = pair
                idx_set.add(pair)

        return order

    def chooseSubsetsAndTransforms(self, Ts, num_pano_img, order, angle):
        num_images = len(Ts)//3
        angle_per_image, angle_rad= 2*np.pi/num_images, np.deg2rad(angle)#- 2*np.pi(angle>180)
        orientation = angle_rad/angle_per_image+0.5
        ref = int(orientation)
        
        # print(ref, angle_per_image, angle_rad)
        odd = num_pano_img%2

        
        ref +=num_images

        if odd:
            offset = num_pano_img // 2  # Offset to pick images on both sides of the reference
            # Subset1: Take images from the left of the reference
            subset1 = order[ref-offset:ref+1][::-1]
            Ts1 = Ts[ref-offset:ref][::-1]
            
            subset2 = order[ref:ref+offset+1]
            Ts2 = Ts[ref:ref+offset]
            
            
        else:
            right_offset = int(orientation >=0.5)
            offset = num_pano_img // 2
            
            subset1 = order[ref-offset+1:ref+1][::-1]
            Ts1 = Ts[ref-offset+1:ref][::-1]
            
            subset2 = order[ref:ref+offset+right_offset]
            Ts2 = Ts[ref:ref+offset]

        return subset1, subset2, Ts1, Ts2
    
    def first_thread(self,batchFlagPosition, imageCount, batchDataPosition, imageWidth, imageHeight,
                    processedFlagPosition, processedDataPosition, processedImageWidth, processedImageHeight, debug = False):
        """""
        This method read the images coming from the software. They have two shared memory files to store the images and the panorama.
        """""
        # global shared_images
        processedImageSize = processedImageHeight*processedImageWidth*3
        imageSize = imageWidth*imageHeight*3
        batchMMF = mmap.mmap(-1, batchDataPosition + imageCount * imageSize, "BatchSharedMemory")
        processedMMF = mmap.mmap(-1, processedDataPosition + processedImageSize, "ProcessedImageSharedMemory")

        while True:
            images, images_bool = readMemory(batchMMF, batchFlagPosition, imageCount, batchDataPosition, imageSize, imageWidth, imageHeight)
            
            droneImInd = np.arange(0, images_bool.shape[0])[images_bool]
            print(f"Index of the drone images to stitch: {droneImInd}")
            with lock:
                
                self.shared_images = images
                self.shared_images_bool = images_bool

            if not self.panoram_queue.empty():
                panorama = self.panoram_queue.get()
                H, W, _ = panorama.shape
                if H != processedImageHeight or W != processedImageWidth:
                    print("shape problem")
                    
                    # panorama = cv2.resize(panorama, (processedImageWidth, processedImageHeight))
                if debug:
                    self.panoram_queue.put(panorama)
                    break
                
                write_memory(processedMMF, processedFlagPosition, processedDataPosition, processedImageSize, panorama)
                del panorama

            # time.sleep(0.005)
            if debug:
                break

    def compute_homographies_and_order(self, keypoints, matches_info, partial_order):
        """""
        COmpute homographies between each best pairs. For n images in the 360 degrees panorama, we have n homographies to compute
        because the last or the first image should be associated with two homographies.
        """""
        num_images= len(keypoints)
        matches_lookup = {(match['image1_index'], match['image2_index']): match['matches'] for match in matches_info}
        
        Hs = np.zeros((num_images, 3, 3))
        for i in range(num_images):
            if i<num_images-1:
                idx1, idx2 = partial_order[i], partial_order[i + 1]
            else:
                idx1, idx2 = partial_order[i], partial_order[0]

            if (idx1, idx2) in matches_lookup:
                matches = matches_lookup[(idx1, idx2)]
                src_p = np.float32([keypoints[idx1][m.queryIdx] for m in matches]).reshape(-1, 2)
                dst_p = np.float32([keypoints[idx2][m.trainIdx] for m in matches]).reshape(-1, 2)
            else:
                matches = matches_lookup[(idx2, idx1)]
                src_p = np.float32([keypoints[idx1][m.trainIdx] for m in matches]).reshape(-1, 2)
                dst_p = np.float32([keypoints[idx2][m.queryIdx] for m in matches]).reshape(-1, 2)

            if dst_p.shape[0]>4:
                Hs[i], _ = cv2.findHomography(dst_p, src_p, method=cv2.RANSAC, ransacReprojThreshold=5, confidence=0.995)

        w, h = self.camera_matrix[:2, -1]*2

        middle_pixel = np.array([w/2, h/2, 1])
        new_middle_pixel = Hs[0]@middle_pixel
        diff_pos = new_middle_pixel[0]/new_middle_pixel[2]-middle_pixel[0]

        if diff_pos<0:
            Hs = np.concatenate((Hs[:1], Hs[1:][::-1]))
            order = np.concatenate(([partial_order[0]], partial_order[1:][::-1]))
            return Hs, order, True
            
        return Hs, partial_order, False
    
    def second_thread(self, front_image_index=0, debug= False):
        """""
        This method uses some of the above methods to extract the order and the homographies of the paired images.
        Input:
            - images: list of NDArrays.
            - front_image_index: the index of the front image of the pilot
        """""
        # global shared_images
        while True:
            with lock:
                if self.shared_images is None:
                    time.sleep(0.1)
                    continue
                else:
                    images = self.shared_images

            if self.warp_type == "cylindrical":
                outputs, ratios, images = self.SP_inference_fast(images)
            else:
                outputs, ratios = self.SP_inference_fast(images)

            keypoints, descriptors = self.keep_best_keypoints(outputs, ratios)
            matches_info, confidences = self.compute_matches_and_confidences(descriptors, keypoints)
            best_pairs = self.find_top_pairs(confidences)
            partial_order = find_cycle_for_360_panorama(confidences, front_image_index, False)[:-1]
            Hs, order, inverted = self.compute_homographies_and_order(keypoints, matches_info, partial_order)

            # put everything in the queues
            self.order_queue.put(order)
            self.homography_queue.put(Hs)
            self.direction_queue.put(inverted)

            print("New parameters given")
            
            if debug:
                return keypoints, Hs, order, inverted, best_pairs, matches_info, confidences, images

    def compose_with_ref(self, images, Hs1, Hs2, subset1, subset2, inverted, clip_x = 8000, clip_y = 2000):
        
        # Initial dimensions of the first image
        h, w = images[0].shape[:2]

        # Initial corners of the reference image
        # corners = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]], dtype=np.float32).reshape(-1, 1, 2)
        corners = np.array([[0, w-1 , w -1, 0],
                              [0, 0, h-1 , h-1 ],
                              [1, 1, 1, 1]], dtype=np.float32)

        if inverted:
            Hs2 = invert_matrices(Hs2)
        else:
            Hs1 = invert_matrices(Hs1)

        # First, apply homographies for subset1 (left side)
        warped_corners_1, H1_acc = apply_homographies(Hs1, corners)
        
        # Then, apply homographies for subset2 (right side)
        warped_corners_2, H2_acc = apply_homographies(Hs2, corners)

        # Calculate the bounding box for the entire panorama
        all_corners = np.concatenate((warped_corners_1, warped_corners_2), axis=1)

        x_min, x_max = np.int32(all_corners[0, :].min()), np.int32(all_corners[0, :].max())
        y_min, y_max = np.int32(all_corners[1, :].min()), np.int32(all_corners[1, :].max())

        print(x_min, x_max, y_min, y_max)

        x_min, x_max =  max(x_min, -clip_x),  min(x_max, clip_x)
        y_min, y_max = max(y_min, -clip_y),  min(y_max, clip_y)

        print(x_min, x_max, y_min, y_max)

        translation_matrix = np.array([[1, 0, -x_min],
                                    [0, 1, -y_min],
                                    [0, 0, 1]], dtype=np.float32)

        panorama_size = (x_max - x_min, y_max - y_min)

        # Warp the reference image and place it on the panorama canvas
        panorama = cv2.warpPerspective(images[subset2[0]], translation_matrix, panorama_size)

        # Warp and blend images from subset1 (left side), skipping the reference image
        for i in range(Hs1.shape[0]):
            H_translate = np.dot(translation_matrix, H1_acc[i])
            
            warped_img = cv2.warpPerspective(images[subset1[i + 1]], H_translate, panorama_size)
            mask = (warped_img > 0).astype(np.uint8)
            panorama[mask > 0] = warped_img[mask > 0]

        # Warp and blend images from subset2 (right side), skipping the reference image
        for i in range(Hs2.shape[0]):
            H_translate = np.dot(translation_matrix, H2_acc[i])
            
            warped_img = cv2.warpPerspective(images[subset2[i + 1]], H_translate, panorama_size)
            mask = (warped_img > 0).astype(np.uint8)
            panorama[mask > 0] = warped_img[mask > 0]

        return panorama
    
    def third_thread(self, headANgle=100, num_pano_img=3, debug=False):
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
        # global shared_images

        order, Hs, inverted = None, None, None
        while True:
            if not self.homography_queue.empty():
                del order, Hs, inverted
                order = self.order_queue.get()
                Hs = self.homography_queue.get()
                inverted = self.direction_queue.get()
                order = np.hstack((order, order, order))
                Hs = np.concatenate((Hs, Hs, Hs))
            elif order is None:
                continue
            
            with lock:
                if self.shared_images is None:
                    time.sleep(0.1)
                    continue
                else:
                    images = self.shared_images

            if self.warp_type == "cylindrical":
                images = [self.cylindricalWarp(img) for img in images]
            
            
            subset1, subset2, Hs1, Hs2 = self.chooseSubsetsAndTransforms(Hs, num_pano_img, order, headANgle)
            pano = self.compose_with_ref(images, Hs1, Hs2, subset1, subset2, inverted)
            # pano2 = self.compose_with_ref_affine(images, M1, M2, subset1, subset2, inverted)
            
            self.panoram_queue.put(pano)

            print("Panorama Given")
            if debug:
                if inverted:
                    print("inverted")
                print(f"Order of the right : {subset2}")
                print(f"Order of the left : {subset1}")

                break


def readMemory(batchMMF, batchFlagPosition, imageCount, batchDataPosition, imageSize, imageWidth, imageHeight):
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
            bb = batchMMF.read(imageCount)
            boolean_list = [bool(b) for b in bb]
            boolean_array=np.array(boolean_list)

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

            return images, boolean_array
    
def write_memory(processedMMF, processedFlagPosition, processedDataPosition, processedImageSize, image_data):
    while True:
        # Read the flag to check if Unity has written new images
        processedMMF.seek(processedFlagPosition)
        flag = struct.unpack('i', processedMMF.read(4))[0]
        
        if flag == 0:  # Unity isn't writing new images
            # Reset flag to 0, indicating we've read the images
            processedMMF.seek(processedFlagPosition)
            processedMMF.write(struct.pack('i', 1))
            
            image_bytes = image_data.tobytes()
            if len(image_bytes) != processedImageSize:
                raise ValueError(f"Image size mismatch: expected {processedImageSize}, got {len(image_bytes)}")
            # Write image
            processedMMF.seek(processedDataPosition)
            processedMMF.write(image_bytes)
            # Convert the byte array into a numpy array

            # Reset flag to 0, indicating we've written the images
            processedMMF.seek(processedFlagPosition)
            processedMMF.write(struct.pack('i', 0))
            break
    # time.sleep(0.01)

def display_images(images):
    for image in images:
        cv2.imshow('Image', image)
        if cv2.waitKey(500) & 0xFF == ord('q'):  # Display each image for 1.5 seconds
            break
    cv2.destroyAllWindows()

def test_reading_writing():
    # Constants (must match the Unity script)
    batchFlagPosition = 0
    imageCount = 16
    batchDataPosition = 1+imageCount
    imageWidth = 240
    imageHeight = 240
    imageSize = imageWidth * imageHeight * 3  # RGB image size
    totalBatchSize = batchDataPosition + imageCount * imageSize

    processedFlagPosition = 0
    processedDataPosition = 4
    processedImageHeight = 240
    processedImageWidth = 240
    processedImageSize = processedImageHeight * processedImageWidth * 3
    totalProcessedSize = processedDataPosition + processedImageSize

    batchMMF = mmap.mmap(-1, totalBatchSize, "BatchSharedMemory")
    processedMMF = mmap.mmap(-1, totalProcessedSize, "ProcessedImageSharedMemory")

    while True:
        images, images_bool = readMemory(batchMMF, batchFlagPosition, imageCount, batchDataPosition, imageSize, imageWidth, imageHeight)
        # print(f"Boolean list: {images_bool}")
        # print(f"Number of image on the borders: {len(images)}")
        droneImInd = np.arange(0, images_bool.shape[0])[images_bool]
        print(f"Index of the drone images to stitch: {droneImInd}")
        if images:
            # print("Images received, displaying first image.")
            t1=time.time()
            write_memory(processedMMF, processedFlagPosition, processedDataPosition, processedImageSize, cv2.flip(images[4], 0))
            print(time.time()-t1)
        else:
            print("No new images, retrying...")

        time.sleep(0.05)  # Sleep briefly before retrying

def test_stitcher():
    # Path to the folder containing calibration images
    images_dir = 'With_calibrated_cam'

    # Find images in the folder
    image_names = glob.glob(os.path.join(images_dir, '*.jpg'))
    images = [cv2.imread(image_name) for image_name in image_names]
    f=4000
    cam_mat = np.array([
        [f, 0, 2.0e+03],  # fy -> fx, cy -> cx
        [0, f, 1.5e+03],  # fx -> fy, cx -> cy
        [0, 0, 1]
    ], dtype=np.float32)

    stitcher = custom_stitcher_SP(camera_matrix=cam_mat, warp_type="cylindrical", algorithm=1, 
                                  trees=5, checks=50, ratio_thresh=0.9, score_threshold=0.1, device="cuda")
    
    stitcher.shared_images = images

    print("Second thread")
    _, _, _, _, _, _, images = stitcher.second_thread(front_image_index=0, debug=True)
    print("Third thread")
    stitcher.third_thread(angle = 70, num_pano_img=3, debug=True)
    print("FInished stitching")

    for image in images:
        cv2.imshow("images after second thread", image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    panorama_resized = cv2.resize(stitcher.panoram_queue.get(), dsize=None, fx=0.15, fy=0.15)
    cv2.imshow("panorama", panorama_resized)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


def test_threading():
    # Path to the folder containing calibration images
    images_dir = 'With_calibrated_cam'

    # Find images in the folder
    image_names = glob.glob(os.path.join(images_dir, '*.jpg'))
    images = [cv2.imread(image_name) for image_name in image_names]
    f=4000
    cam_mat = np.array([
        [f, 0, 2.0e+03],  # fy -> fx, cy -> cx
        [0, f, 1.5e+03],  # fx -> fy, cx -> cy
        [0, 0, 1]
    ], dtype=np.float32)

    stitcher = custom_stitcher_SP(camera_matrix=cam_mat, warp_type="cylindrical", algorithm=1, 
                                  trees=5, checks=50, ratio_thresh=0.9, score_threshold=0.1, device="cuda")
    
    def simFirstThread(imgs):
        t0 = time.time()
        while True:
            with lock:
                stitcher.shared_images = imgs

            if not stitcher.panoram_queue.empty():
                panorama = stitcher.panoram_queue.get()
                #Simulate the writing with a small time sleep (to be ckecked)
                time.sleep(0.003)
                print(f"panorama obtained in {time.time()-t0}")
                t0 = time.time()
                # write_memory(processedMMF, processedFlagPosition, processedDataPosition, processedImageSize, panorama)
            time.sleep(0.01)

    # Start a thread for simulating the image feed
    first_thread = threading.Thread(target=simFirstThread, args=(images,))
    first_thread.daemon = True
    first_thread.start()

    sec_thread = threading.Thread(target=stitcher.second_thread, args=(0, False))
    sec_thread.daemon = True
    sec_thread.start()

    third_thread = threading.Thread(target=stitcher.third_thread, args=(100, 3, False))
    third_thread.daemon = True
    third_thread.start()

    while True:
        time.sleep(10)

def test_one_image(front_image_index, angle):
    batchFlagPosition = 0
    imageCount = 16
    batchDataPosition = 1+imageCount
    imageWidth = 300
    imageHeight = 300

    processedFlagPosition = 0
    processedDataPosition = 4
    processedImageHeight = 240
    processedImageWidth = 240

    f = 160

    cam_mat = np.array([[f, 0, imageWidth/2], 
                        [0, f, imageHeight/2],
                        [0, 0, 1]])

    stitcher = custom_stitcher_SP(camera_matrix=cam_mat, warp_type="cylindrical", algorithm=1, trees=5, checks=50, ratio_thresh=0.7, score_threshold=0.0, device="cuda")

    stitcher.first_thread(batchFlagPosition, imageCount, batchDataPosition, imageWidth, imageHeight,
                    processedFlagPosition, processedDataPosition, processedImageWidth, processedImageHeight, debug=True)
    # stitcher.image_queue.put(images)
    keypoints, Hs, order, inverted, best_pairs, matches_info, confidences, imgs =stitcher.second_thread(front_image_index=front_image_index, debug=True)

    print("order", order)
        

    stitcher.first_thread(batchFlagPosition, imageCount, batchDataPosition, imageWidth, imageHeight,
                    processedFlagPosition, processedDataPosition, processedImageWidth, processedImageHeight, debug=True)
    # stitcher.image_queue.put(images)
    stitcher.third_thread(headANgle=angle, debug = True)
    stitcher.first_thread(batchFlagPosition, imageCount, batchDataPosition, imageWidth, imageHeight,
                    processedFlagPosition, processedDataPosition, processedImageWidth, processedImageHeight, debug=True)

    panorama = stitcher.panoram_queue.get()
    panorama_resized = cv2.resize(panorama, dsize=None, fx=0.5, fy=0.5)

    # print(f"Original panorama shape{panorama.shape}")
    # print(f"Plot panorama of shape{panorama_resized.shape}")

    cv2.imshow("panorama", panorama_resized)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    cv2.imwrite("result_3_images.png", panorama_resized)


def main():

    batchFlagPosition = 0
    imageCount = 16
    batchDataPosition = 1+imageCount
    imageWidth = 240
    imageHeight = 240
    imageSize = imageWidth * imageHeight * 3  # RGB image size

    processedFlagPosition = 0
    processedDataPosition = 4
    processedImageHeight = 240
    processedImageWidth = 240

    f = 1000

    cam_mat = np.array([[f, 0, imageWidth/2], 
                        [0, f, imageHeight/2],
                        [0, 0, 1]])

    stitcher = custom_stitcher_SP(camera_matrix=cam_mat, warp_type="cylindrical", algorithm=1, trees=5, checks=50, ratio_thresh=0.9, score_threshold=0.1, device="cuda")

    # Start a thread for simulating the image feed
    first_thread = threading.Thread(target=stitcher.first_thread, args=(batchFlagPosition, imageCount, batchDataPosition, imageSize, imageHeight, imageWidth,
                    processedFlagPosition, processedDataPosition, processedImageHeight, processedImageWidth))
    first_thread.daemon = True
    first_thread.start()

    sec_thread = threading.Thread(target=stitcher.second_thread, args=(0, False))
    sec_thread.daemon = True
    sec_thread.start()

    third_thread = threading.Thread(target=stitcher.third_thread, args=(100, 3, False))
    third_thread.daemon = True
    third_thread.start()
    

if __name__ == '__main__':
    
    # main function, the final result
    # main()

    # Test reading the images from shared memory and write an image in the panorama memory
    # test_reading_writing()

    # Test stitcher with own images
    # test_stitcher()

    # Test threading function
    # test_threading()

    # Test one stitched camera
    front_image_index = 0
    angle = 344
    
    test_one_image(front_image_index, angle)