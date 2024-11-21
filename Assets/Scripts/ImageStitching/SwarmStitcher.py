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
# from scipy.optimize import minimize


lock = threading.Lock()

global processedImageWidth
global processedImageHeight

def smooth_homography(H_new, H_prev, alpha=0.8):
    """
    Smooth the new homography matrix by blending it with the previous one.
    :param H_new: New homography matrix (3x3).
    :param H_prev: Previous homography matrix (3x3).
    :param alpha: Smoothing factor (0 < alpha <= 1). Higher values favor the previous matrix.
    :return: Smoothed homography matrix.
    """
    return alpha * H_prev + (1 - alpha) * H_new

def weighted_average_homography(H_new, H_prev, weight=0.1):
    """
    Weighted averaging of homography matrices.
    :param H_new: Newly computed homography matrix.
    :param H_prev: Previous homography matrix.
    :param weight: Weight given to the previous homography.
    :return: Smoothed homography matrix.
    """
    delta = np.linalg.norm(H_new - H_prev, ord='fro')
    adaptive_weight = weight / (1 + delta)  # Higher weight if the change is small
    return adaptive_weight * H_prev + (1 - adaptive_weight) * H_new

# def compute_regularized_homography(points1, points2, H_prev, lambda_reg=0.1):
#     """
#     Compute homography matrix with regularization using optimization.
#     :param points1: Source points (Nx2).
#     :param points2: Destination points (Nx2).
#     :param H_prev: Previous homography matrix.
#     :param lambda_reg: Regularization weight.
#     :return: Regularized homography matrix.
#     """

#     def reprojection_error(H_flat, points1, points2, H_prev, lambda_reg):
#         H = H_flat.reshape(3, 3)
#         points1_homog = np.hstack((points1, np.ones((points1.shape[0], 1))))
#         points2_proj = (H @ points1_homog.T).T
#         points2_proj /= points2_proj[:, 2][:, None]
#         reprojection_error = np.sum(np.linalg.norm(points2_proj[:, :2] - points2, axis=1)**2)
#         regularization_term = lambda_reg * np.linalg.norm(H - H_prev, ord='fro')**2
#         return reprojection_error + regularization_term

#     # Initial guess: the previous homography
#     H_init = H_prev.flatten()

#     # Minimize the reprojection error with regularization
#     result = minimize(
#         reprojection_error, 
#         H_init, 
#         args=(points1, points2, H_prev, lambda_reg), 
#         method='BFGS'
#     )

#     H_optimized = result.x.reshape(3, 3)
#     return H_optimized


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

def invert_affine_matrices(Ms):
    Ms_inverted = np.zeros_like(Ms)
    num_mat = Ms_inverted.shape[0]
    for i in range(num_mat):
        Ms_inverted[i]=cv2.invertAffineTransform(Ms[i])
    return Ms_inverted

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

def apply_affine_matrices(Ms, corners):
    """
    Apply the affine matrices to image corners and return the transformed corners.
    Optimized for affine transformations, assuming 2x3 matrices.
    """

    num_images, num_corners = Ms.shape[0], 4
    tot_points = num_corners * num_images
    all_corners = np.zeros((2, tot_points), dtype=np.float32)  # Store 2D points
    M_accum = np.zeros((num_images, 2, 3), dtype=np.float32)  # Accumulated affine matrices

    M = np.eye(2,3, dtype=np.float32)  # Start with identity affine matrix

    for i in range(num_images):
        # Convert the affine matrix to a 3x3 homography-like matrix for accumulation
        M_curr = np.vstack([Ms[i], [0, 0, 1]])  # Convert to 3x3 by adding [0, 0, 1] row
        M_accum_3x3 = np.dot(M_curr, np.vstack([M, [0, 0, 1]]))  # Update accumulated affine matrix
        
        # Extract the 2x3 part for affine transformation (discard the 3rd row/column)
        M_accum[i] = M_accum_3x3[:2, :]
        
        # Apply the current affine matrix to the corners
        new_corners = np.dot(M_accum[i], corners)  # Only need 2xN points for affine
        all_corners[:, i*num_corners: (i+1)*num_corners] = new_corners

    print(M_accum)
    
    return all_corners, M_accum

@jit(nopython=True) 
def invert_matrices(Hs:np.ndarray)->np.ndarray:
    Hs_inverted = np.zeros_like(Hs)
    num_mat = Hs_inverted.shape[0]
    for i in range(num_mat):
        if np.abs(np.linalg.det(Hs[i])) > 1e-10:
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
        self.headAngle = 30

    def ORB_extraction(self, images):

        orb = cv2.ORB_create()
        keypoints = []
        descriptors = []

        for image in images:
            gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            kpts, dpts = orb.detectAndCompute(gray_img, None)

            if kpts is not None and dpts is not None:
                keypoints.append(np.array([kp.pt for kp in kpts]))
                descriptors.append(dpts.astype(np.float32))
            else:
                keypoints.append(np.array([]))
                descriptors.append(np.array([]))

        return keypoints, descriptors

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

                # matches = self.FLANN_matching(desc1, desc2)
                matches = self.BF_matching(desc1, desc2)
                num_matches = len(matches)

                H1, H2 = None, None

                # It was originally made to compute Homographies (needs 4 matches at least to be computed)
                # This should be removed
                if num_matches> 4:
                    src_p = np.float32([keypoints[i][m.queryIdx] for m in matches]).reshape(-1, 2)
                    dst_p = np.float32([keypoints[j][m.trainIdx] for m in matches]).reshape(-1, 2)
                    H1, mask1 = cv2.findHomography(dst_p, src_p, cv2.RANSAC, ransacReprojThreshold=5)
                    H2, mask2 = cv2.findHomography(src_p, dst_p, cv2.RANSAC, ransacReprojThreshold=5)
                    max_inliers = max(np.sum(mask1.ravel()), np.sum(mask2.ravel()))
                    conf = max_inliers / (8 + 0.3 * num_matches)

                    # This is not the exact same equation as the one given by the opencv
                    # function. This could be removed and just plug the num_matches in 
                    # the confidence matrix 
                    # conf = num_matches#num_matches / (8 + 0.3 * num_matches)

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
        angle -= angle//360 *360
        if angle<0:
            angle+=360
        angle_per_image, angle_rad= 2*np.pi/num_images, np.deg2rad(angle)#- 2*np.pi(angle>180)
        orientation = angle_rad/angle_per_image+0.5
        ref = int(orientation)
        
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

    def compute_homographies_and_order(self, keypoints, matches_info, partial_order, H_prev=None):
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
                Hs[i], _ = cv2.findHomography(dst_p, src_p, method=cv2.RANSAC, ransacReprojThreshold=3, confidence=0.995)
                # Hs[i], _ = cv2.findHomography(dst_p, src_p, method=0, ransacReprojThreshold=1, confidence=0.995)

                # if H_prev is not None:
                #     Hs[i]=compute_regularized_homography(src_p, dst_p, H_prev[i], lambda_reg=0.5)

        w, h = self.camera_matrix[:2, -1]*2

        middle_pixel = np.array([w/2, h/2, 1])
        new_middle_pixel = Hs[0]@middle_pixel
        diff_pos = new_middle_pixel[0]/new_middle_pixel[2]-middle_pixel[0]

        if diff_pos<0:
            Hs = np.concatenate((Hs[:1], Hs[1:][::-1]))
            order = np.concatenate(([partial_order[0]], partial_order[1:][::-1]))
            return Hs, order, True
            
        return Hs, partial_order, False
    
    def compute_affines_and_order(self, keypoints, matches_info, partial_order):
        """""
        COmpute homographies between each best pairs. For n images in the 360 degrees panorama, we have n homographies to compute
        because the last or the first image should be associated with two homographies.
        """""
        num_images= len(keypoints)
        matches_lookup = {(match['image1_index'], match['image2_index']): match['matches'] for match in matches_info}
        
        Hs = np.zeros((num_images, 2, 3))
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
                Hs[i], _ = cv2.estimateAffine2D(dst_p, src_p, method=cv2.RANSAC, ransacReprojThreshold=5, confidence=0.995)

        w, h = self.camera_matrix[:2, -1]*2

        middle_pixel = np.array([w/2, h/2, 1])
        new_middle_pixel = Hs[0]@middle_pixel
        diff_pos = new_middle_pixel[0]-middle_pixel[0]

        if diff_pos<0:
            Hs = np.concatenate((Hs[:1], Hs[1:][::-1]))
            order = np.concatenate(([partial_order[0]], partial_order[1:][::-1]))
            return Hs, order, True
            
        return Hs, partial_order, False

    def affineStitching(self, images, Ms1, Ms2, subset1, subset2, inverted, clip_x = 8000, clip_y = 2000):
        
        # Initial dimensions of the first image
        h, w = images[0].shape[:2]

        # Initial corners of the reference image
        corners = np.array([[0, w-1 , w -1, 0],
                              [0, 0, h-1 , h-1 ],
                              [1, 1, 1, 1]], dtype=np.float32)

        if inverted:
            Ms2 = invert_affine_matrices(Ms2)
        else:
            Ms1 = invert_affine_matrices(Ms1)

        # First, apply affine transformations for subset1 (left side)
        warped_corners_1, M1_acc = apply_affine_matrices(Ms1, corners)

        # Then, apply affine transformations for subset2 (right side)
        warped_corners_2, M2_acc = apply_affine_matrices(Ms2, corners)

        # Calculate the bounding box for the entire panorama
        all_corners = np.concatenate((warped_corners_1, warped_corners_2), axis=1)

        x_min, x_max = np.int32(all_corners[0, :].min()), np.int32(all_corners[0, :].max())
        y_min, y_max = np.int32(all_corners[1, :].min()), np.int32(all_corners[1, :].max())

        # print(x_min, x_max, y_min, y_max)

        x_min, x_max = max(x_min, -clip_x), min(x_max, clip_x)
        y_min, y_max = max(y_min, -clip_y), min(y_max, clip_y)

        # print(x_min, x_max, y_min, y_max)

        # Translation matrix for panorama placement
        translation_matrix = np.array([[1, 0, -x_min],
                                    [0, 1, -y_min]], dtype=np.float32)

        panorama_size = (x_max - x_min, y_max - y_min)

        # Warp the reference image and place it on the panorama canvas using cv2.warpAffine
        panorama = cv2.warpAffine(images[subset2[0]], translation_matrix, panorama_size)
        M1_acc[:,:2,-1]+=translation_matrix[:2,-1]
        # M2_acc[:,:2,-1]+=translation_matrix[:2,-1]

        # Warp and blend images from subset1 (left side), skipping the reference image
        for i in range(M1_acc.shape[0]):
            M_translate = M1_acc[i]

            warped_img = cv2.warpAffine(images[subset1[i + 1]], M_translate, panorama_size)

            mask = (warped_img > 0).astype(np.uint8)
            panorama[mask > 0] = warped_img[mask > 0]

        # Warp and blend images from subset2 (right side), skipping the reference image
        # for i in range(M2_acc.shape[0]):
        #     M_translate = M2_acc[i]

        #     warped_img = cv2.warpAffine(images[subset2[i + 1]], M_translate, panorama_size)

        #     mask = (warped_img > 0).astype(np.uint8)
        #     panorama[mask > 0] = warped_img[mask > 0]

        return panorama
    
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

        # print(x_min, x_max, y_min, y_max)

        x_min, x_max =  max(x_min, -clip_x),  min(x_max, clip_x)
        y_min, y_max = max(y_min, -clip_y),  min(y_max, clip_y)

        # print(x_min, x_max, y_min, y_max)

        # translation_matrix = np.array([[1, 0, -x_min],
        #                             [0, 1, -y_min],
        #                             [0, 0, 1]], dtype=np.float32)

        panorama_width = x_max - x_min
        panorama_height = y_max - y_min

        center_x_offset = panorama_width // 2 - w // 2
        center_y_offset = panorama_height // 2 - h // 2

    # Translation matrix to center the reference image
        translation_matrix = np.array([[1, 0, center_x_offset],
                                    [0, 1, center_y_offset],
                                    [0, 0, 1]], dtype=np.float32)

        panorama_size = (panorama_width, panorama_height)

        # Warp the reference image and place it on the panorama canvas
        panorama = cv2.warpPerspective(images[subset2[0]], translation_matrix, panorama_size)
        ref_mask = (panorama > 0).astype(np.uint8)

        # Warp and blend images from subset1 (left side), skipping the reference image
        for i in range(Hs1.shape[0]):
            H_translate = np.dot(translation_matrix, H1_acc[i])
            
            print(len(images), i, subset1[i + 1])
            warped_img = cv2.warpPerspective(images[subset1[i + 1]], H_translate, panorama_size)

            mask = (warped_img > 0).astype(np.uint8)
            panorama[(mask > 0) & (ref_mask == 0) ] = warped_img[(mask > 0) & (ref_mask == 0)]

        # Warp and blend images from subset2 (right side), skipping the reference image
        for i in range(Hs2.shape[0]):
            print(len(images), i, subset1[i + 1])
            H_translate = np.dot(translation_matrix, H2_acc[i])
            warped_img = cv2.warpPerspective(images[subset2[i + 1]], H_translate, panorama_size)

            mask = (warped_img > 0).astype(np.uint8)
            panorama[(mask > 0) & (ref_mask == 0)] = warped_img[(mask > 0) & (ref_mask == 0)]

        # _, thresh = cv2.threshold(cv2.cvtColor(panorama, cv2.COLOR_RGB2GRAY), 1, 255, cv2.THRESH_BINARY)
        # # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # x, y, w, h = cv2.boundingRect(thresh)#cv2.boundingRect(contours[0])

        return panorama#[y:y+h, x:x+w]
    
    def compose_with_defined_size(self, images, Hs1, Hs2, subset1, subset2, inverted,  panoWidth=500, panoHeight=400):
        
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
        _, H1_acc = apply_homographies(Hs1, corners)
        
        # Then, apply homographies for subset2 (right side)
        _, H2_acc = apply_homographies(Hs2, corners)

        center_x_offset = panoWidth // 2 - w // 2
        center_y_offset = panoHeight // 2 - h // 2

    # Translation matrix to center the reference image
        translation_matrix = np.array([[1, 0, center_x_offset],
                                    [0, 1, center_y_offset],
                                    [0, 0, 1]], dtype=np.float32)

        panorama_size = (int(panoWidth), int(panoHeight))

        # Warp the reference image and place it on the panorama canvas
        panorama = cv2.warpPerspective(images[subset2[0]], translation_matrix, panorama_size)
        ref_mask = (panorama > 0).astype(np.uint8)

        # Warp and blend images from subset1 (left side), skipping the reference image
        for i in range(Hs1.shape[0]):
            H_translate = np.dot(translation_matrix, H1_acc[i])
            
            warped_img = cv2.warpPerspective(images[subset1[i + 1]], H_translate, panorama_size)

            mask = (warped_img > 0).astype(np.uint8)
            panorama[(mask > 0) & (ref_mask == 0) ] = warped_img[(mask > 0) & (ref_mask == 0)]

        # Warp and blend images from subset2 (right side), skipping the reference image
        for i in range(Hs2.shape[0]):
            H_translate = np.dot(translation_matrix, H2_acc[i])
            warped_img = cv2.warpPerspective(images[subset2[i + 1]], H_translate, panorama_size)

            mask = (warped_img > 0).astype(np.uint8)
            panorama[(mask > 0) & (ref_mask == 0)] = warped_img[(mask > 0) & (ref_mask == 0)]

        # _, thresh = cv2.threshold(cv2.cvtColor(panorama, cv2.COLOR_RGB2GRAY), 1, 255, cv2.THRESH_BINARY)
        # # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # x, y, w, h = cv2.boundingRect(thresh)#cv2.boundingRect(contours[0])

        return panorama#[y:y+h, x:x+w]

    def first_thread(self, debug = False):
        """""
        This method read the images coming from the software. They have three shared memory files to store the images, the panorama and the metadatas.
        """""
        global processedImageWidth
        global processedImageHeight

        flagPosition = 0
        processedDataPosition = 4
        # global shared_images
        metadataSize = 20 + 64 + 1 # 20 bytes for ints (5x4 bytes) + 64 bytes for string + 1 byte bool
        metadataMMF = mmap.mmap(-1, metadataSize, "MetadataSharedMemory")

        # Read first time metadata to initialize the memories:
        output = readMetadataMemory(metadataMMF)
        typeOfStitcher, isCylindrical= output["string"], output["boolean"]
        
        batchImageWidth, batchImageHeight, imageCount, processedImageWidth, processedImageHeight= output["int_values"]
        
        imageSize, headANglePosition, batchDataPosition, processedImageSize = UpdateValues(batchImageWidth, batchImageHeight, imageCount, processedImageWidth, processedImageHeight)
        batchMMF = mmap.mmap(-1, batchDataPosition +  imageCount* imageSize, "BatchSharedMemory")
        processedMMF = mmap.mmap(-1, processedDataPosition + processedImageSize, "ProcessedImageSharedMemory")

        first_loop = True
        while True:

            ### New part
            output = readMetadataMemory(metadataMMF)
            typeOfStitcher, isCylindrical= output["string"], output["boolean"]
            
            batchImageWidth, batchImageHeight, imageCount, processedImageWidth, processedImageHeight= output["int_values"]
            
            imageSize, headANglePosition, batchDataPosition, processedImageSize = UpdateValues(batchImageWidth, batchImageHeight, imageCount, processedImageWidth, processedImageHeight)
            
            # print("batchImageWidth", batchImageWidth)
            # print("batchImageHeight", batchImageHeight)
            # print("imageCount", imageCount)
            # print("processedImageWidth", processedImageWidth)
            # print("processedImageHeight", processedImageHeight)
            # print("processedImageSize", processedImageSize)
            # print("imageSize", imageSize)
            # print("batchDataPosition", batchDataPosition)

            batchMMF = mmap.mmap(-1, batchDataPosition +  imageCount* imageSize, "BatchSharedMemory")
            processedMMF = mmap.mmap(-1, processedDataPosition + processedImageSize, "ProcessedImageSharedMemory")

            print(f"Type of STitcher: {typeOfStitcher}.\n Is cylindrical: {isCylindrical}")
            ###

            try:
                images, images_bool, self.headAngle = readMemory(batchMMF, flagPosition, headANglePosition, imageCount, batchDataPosition, imageSize, batchImageWidth, batchImageHeight)
            except:
                print("problem")
                continue
            
            # droneImInd = np.arange(0, images_bool.shape[0])[images_bool]
            # print(f"Index of the drone images to stitch: {droneImInd}")
            with lock:
                self.shared_images = images
                self.shared_images_bool = images_bool

            if not self.panoram_queue.empty():
                panorama = self.panoram_queue.get()
                H, W, _ = panorama.shape
                if H != processedImageHeight or W != processedImageWidth:
                    try:
                        panorama = cv2.resize(panorama, (processedImageWidth, processedImageHeight))
                    except:
                        continue
                # if debug:
                #     self.panoram_queue.put(panorama)
                #     break
                try:
                    write_memory(processedMMF, flagPosition, processedDataPosition, processedImageSize, cv2.flip(panorama, 0))
                    del panorama
                except:
                    continue


            time.sleep(0.05)

            if first_loop:
                first_loop = False
                time.sleep(1.)

            if debug:
                break

    def second_thread(self, front_image_index=0, debug= False):
        """""
        This method uses some of the above methods to extract the order and the homographies of the paired images.
        Input:
            - images: list of NDArrays.
            - front_image_index: the index of the front image of the pilot
        """""
        Hs = None
        # global shared_images
        while True:
            if self.shared_images is None:
                    print("Second thread sleep")
                    time.sleep(0.4)
                    continue
            
            with lock:
                images = self.shared_images

            t = time.time()
            if self.warp_type == "cylindrical":
                outputs, ratios, images = self.SP_inference_fast(images)
            else:
                outputs, ratios = self.SP_inference_fast(images)

            
            keypoints, descriptors = self.keep_best_keypoints(outputs, ratios)
            t1 = time.time()
            # keypoints, descriptors = self.ORB_extraction(images)
            matches_info, confidences = self.compute_matches_and_confidences(descriptors, keypoints)
            t2 = time.time()
            best_pairs = self.find_top_pairs(confidences)
            partial_order = find_cycle_for_360_panorama(confidences, front_image_index, False)[:-1]
            H, order, inverted = self.compute_homographies_and_order(keypoints, matches_info, partial_order, Hs)
            if Hs is not None:
                for i in range(H.shape[0]):
                    Hs[i] = smooth_homography(H[i], Hs[i], alpha=0.08)
            else:
                Hs = H

            t3 = time.time()
            # Ms, order, inverted = self.compute_affines_and_order(keypoints, matches_info, partial_order)

            # print("time to extract keypoints:", t1-t)
            # print("time to compute matches:", t2-t1)
            # print("time to compute homographies:", t3-t2)

            # put everything in the queues
            self.order_queue.put(order)
            self.homography_queue.put(Hs)
            # self.homography_queue.put(Ms)
            self.direction_queue.put(inverted)
            print(order)
            # print(f"Time to compute the Parameters: {time.time()-t}")
            # time.sleep(0.5)
            # print("New parameters given")
            
            if debug:
                return keypoints, Hs, order, inverted, best_pairs, matches_info, confidences, images
                # return keypoints, Ms, order, inverted, best_pairs, matches_info, confidences, images

    def third_thread(self, headAngle=100, num_pano_img=3, debug=False):
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

        global processedImageWidth
        global processedImageHeight

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
                time.sleep(0.4)
                continue
            
            with lock:
                images = self.shared_images

            t = time.time()
            if self.warp_type == "cylindrical":
                images = [self.cylindricalWarp(img) for img in images]
            
            
            subset1, subset2, Hs1, Hs2 = self.chooseSubsetsAndTransforms(Hs, num_pano_img, order, headAngle)
            # pano = self.compose_with_ref(images, Hs1, Hs2, subset1, subset2, inverted)
            pano = self.compose_with_defined_size(images, Hs1, Hs2, subset1, subset2, inverted, panoWidth=processedImageWidth, panoHeight=processedImageHeight)
            # pano = self.affineStitching(images, Hs1, Hs2, subset1, subset2, inverted)
            
            self.panoram_queue.put(pano)

            # print(f"Time to make the Panorama: {time.time()-t}")
            # print("Panorama Given")
            if debug:
                if inverted:
                    print("inverted")
                print(f"Order of the right : {subset2}")
                print(f"Order of the left : {subset1}")

                break

def readMetadataMemory(metadataMMF :mmap )->dict:
    
    # Read the integers
    metadataMMF.seek(0)
    int_values = struct.unpack('iiiii', metadataMMF.read(20))  # 5 integers

    # Read the string
    raw_string = metadataMMF.read(64)
    metadata_string = raw_string.decode('utf-8').rstrip('\x00')  # Remove padding

    # Read the boolean
    raw_bool = metadataMMF.read(1)
    metadata_bool = bool(struct.unpack('B', raw_bool)[0])  # Unpack as unsigned char

    # Return all parsed metadata
    return {
        "int_values": int_values,  # Tuple of 5 integers
        "string": metadata_string,
        "boolean": metadata_bool
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

# Have to add function that will look if parameters have changed and if so, recompute cylindrical warping
# Think about how to change the stitcher. Maybe instead of 3 methods as thread, just do 3 functions

# def hasSmallStretchHomography(H: np.ndarray, corners, tolerance = 2):

#     H[:2, -1] = np.zeros(2)
#     _, newCorners =apply_homographies(np.expand_dims(H, axis=0), corners)

#     if newCorners>tolerance*corners[:2]:
#         print("Too big changes")
#         return False    
    
#     return True

# def hassmallChangeHomography(H_new, H_prev, criterion = 0.5):
#     """
#     Assume same dimensions for H_new and H_prev
#     """
#     return np.linalg.norm(H_prev-H_new, 'fro')>criterion

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

def test_one_image(front_image_index, angle, num_pano_img):
    batchFlagPosition = 0
    imageCount = 16
    batchDataPosition = 1+imageCount
    imageWidth = 300
    imageHeight = 300

    processedFlagPosition = 0
    processedDataPosition = 4
    processedImageHeight = 300
    processedImageWidth = 300

    f = 160

    cam_mat = np.array([[f, 0, imageWidth/2], 
                        [0, f, imageHeight/2],
                        [0, 0, 1]])

    stitcher = custom_stitcher_SP(camera_matrix=cam_mat, warp_type="cylindrica", algorithm=1, trees=5, checks=50, ratio_thresh=0.7, score_threshold=0.0, device="cuda")

    stitcher.first_thread(batchFlagPosition, imageCount, batchDataPosition, imageWidth, imageHeight,
                    processedFlagPosition, processedDataPosition, processedImageWidth, processedImageHeight, debug=True)
    # stitcher.image_queue.put(images)
    keypoints, Hs, order, inverted, best_pairs, matches_info, confidences, imgs =stitcher.second_thread(front_image_index=front_image_index, debug=True)

    print("order", order)

    for i, image in enumerate(imgs):
        if i ==2:
            break
        img = np.array(image)  # Convert PIL image to numpy array
        cv2.imwrite(f"image{i}.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for keypoint in keypoints[i]:
            keypoint_x, keypoint_y = int(keypoint[0]), int(keypoint[1])
            color = tuple([255, 0, 0])
            image = cv2.circle(img, (keypoint_x, keypoint_y), 5, color)

        
        cv2.imwrite(f"keypoints{i}.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    i=0
    for match_info in matches_info:
        if i ==1:
            break

        img1_idx = match_info['image1_index']
        img2_idx = match_info['image2_index']
        matches = match_info['matches']

        img1 = np.array(imgs[img1_idx])
        img2 = np.array(imgs[img2_idx])

        # Convert keypoints to the format expected by cv2.drawMatches
        keypoints1 = [cv2.KeyPoint(x.astype(float), y.astype(float), 1) for x, y in keypoints[img1_idx]]
        keypoints2 = [cv2.KeyPoint(x.astype(float), y.astype(float), 1) for x, y in keypoints[img2_idx]]

        img1_with_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("matches.png", cv2.cvtColor(img1_with_matches, cv2.COLOR_BGR2RGB))

        i+=1
        

    stitcher.first_thread(batchFlagPosition, imageCount, batchDataPosition, imageWidth, imageHeight,
                    processedFlagPosition, processedDataPosition, processedImageWidth, processedImageHeight, debug=True)
    # stitcher.image_queue.put(images)
    stitcher.third_thread(headANgle=angle, num_pano_img= num_pano_img, debug = True)
    stitcher.first_thread(batchFlagPosition, imageCount, batchDataPosition, imageWidth, imageHeight,
                    processedFlagPosition, processedDataPosition, processedImageWidth, processedImageHeight, debug=True)

    panorama = stitcher.panoram_queue.get()
    panorama_resized = panorama#cv2.resize(panorama, dsize=None, fx=1, fy=1)

    # print(f"Original panorama shape{panorama.shape}")
    # print(f"Plot panorama of shape{panorama_resized.shape}")

    # cv2.imshow("panorama", panorama_resized)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()
    panorama_resized = cv2.cvtColor(panorama_resized, cv2.COLOR_BGR2RGB)
    cv2.imwrite("result_2_images_shift.png", panorama_resized)

def main():

    imageWidth = 300
    imageHeight = 300

    f = 160

    cam_mat = np.array([[f, 0, imageWidth/2], 
                        [0, f, imageHeight/2],
                        [0, 0, 1]])

    stitcher = custom_stitcher_SP(camera_matrix=cam_mat, warp_type="cylindrica", algorithm=1, trees=5, checks=50, ratio_thresh=0.7, score_threshold=0.05, device="cuda")

    first_thread = threading.Thread(target=stitcher.first_thread, args=(False,))
    first_thread.daemon = True
    first_thread.start()

    sec_thread = threading.Thread(target=stitcher.second_thread, args=(0, False))
    sec_thread.daemon = True
    sec_thread.start()

    third_thread = threading.Thread(target=stitcher.third_thread, args=(0, 3, False))
    third_thread.daemon = True
    third_thread.start()

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