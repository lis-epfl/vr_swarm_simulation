##################
# OLD VERSION of the StitcherThreading for DL models
##################

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

import sys
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.abspath("UDIS2_main\Warp\Codes"))

from UDIS2_main.Warp.Codes.utils import *

import UDIS2_main.Warp.Codes.grid_res as grid_res
from UDIS2_main.Warp.Codes.network import build_output_model, get_stitched_result, Network, build_new_ft_model
from UDIS2_main.Warp.Codes.loss import cal_lp_loss2

import cv2
import torchvision.transforms as T

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

# @jit(nopython=True, parallel=True)
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

class custom_stitcher:
    def __init__(self, camera_matrix, warp_type="cylindrical", full_cylinder = True, algorithm=1, 
                 trees=5, checks=50, ratio_thresh = 0.7, score_threshold = 0.2, device = "cpu", 
                 model_path="UDIS2_main\Warp"):
        
        # SuperPoint model initialization that is used to extract features from images
        self.model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
        self.model.eval()
        self.device = torch.device(device)
        
        self.H_SP = 480
        self.W_SP = 640
        self.score_threshold = score_threshold

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

        # UDIS parameters   
        self.resize_512 = T.Resize((512,512))
        self.net = Network()
        MODEL_DIR = os.path.join(model_path, 'model')

        ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
        ckpt_list.sort()
        if len(ckpt_list) != 0:
            model_path = ckpt_list[-1]
            checkpoint = torch.load(model_path)
            self.net.load_state_dict(checkpoint['model'])
            print('load model from {}!'.format(model_path))

        self.net.to(self.device)

        # Thread queues
        self.homography_queue = queue.Queue(1)
        self.order_queue = queue.Queue(1)
        self.direction_queue = queue.Queue(1)
        self.panoram_queue = queue.Queue(1)
        self.shared_images = None
        self.shared_images_bool = None

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
        inputs = torch.FloatTensor(np.array(inputs)).permute(0,3,1,2).to(self.model.device)*rescale_factor
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
                    src_p = np.float32([keypoints[i][m.queryIdx] for m in matches]).reshape(-1, 2)
                    dst_p = np.float32([keypoints[j][m.trainIdx] for m in matches]).reshape(-1, 2)
                    H1, mask1 = cv2.findHomography(dst_p, src_p, 0, ransacReprojThreshold=5)
                    H2, mask2 = cv2.findHomography(src_p, dst_p, 0, ransacReprojThreshold=5)
                    max_inliers = max(np.sum(mask1.ravel()), np.sum(mask2.ravel()))
                    conf = max_inliers / (8 + 0.3 * num_matches)

                    # This is not the exact same equation as the one given by the opencv
                    # function. This could be removed and just plug the num_matches in 
                    # the confidence matrix 
                    #conf = num_matches / (8 + 0.3 * num_matches)

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
        
        print(f"confidences:\n {confidences}")

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

    def chooseSubsetsAndTransforms(self, num_pano_img, order, angle):
        num_images = len(order)//3
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
            subset2 = order[ref:ref+offset+1]
            
            
        else:
            right_offset = int(orientation >=0.5)
            offset = num_pano_img // 2
            
            subset1 = order[ref-offset+1:ref+1][::-1]
            subset2 = order[ref:ref+offset+right_offset]

        return subset1, subset2

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

        shiftup2 = np.argmax(right_warp[:,0].mean(axis=1) != 0)
        shiftdown2 = shiftup2 + np.argmax(right_warp[shiftup2:, 0].mean(axis=1) == 0)

        # We have to flip the images to warp the left image and keep central image as the reference
        image1, image2 = cv2.flip(images[subset1[0]], 1), cv2.flip(images[subset1[1]], 1)

        # print("warping left")
        left_warp = self.UDIS_warping(image1, image2)
        
        shiftup1 = np.argmax(left_warp[:,0].mean(axis=1) != 0)
        shiftdown1 = shiftup1 + np.argmax(left_warp[shiftup1:, 0].mean(axis=1) == 0)

        left_warp = cv2.flip(left_warp, 1)

        rightSize = right_warp.shape
        leftSize = left_warp.shape

        diff2x, diff2y = rightSize[1]-w, rightSize[0]-h
        diff1x, diff1y = leftSize[1]-w, leftSize[0]-h

        pano = np.zeros((diff1y+diff2y+h, diff1x+diff2x+w, 3))
        # pano = np.zeros((h, diff1x+diff2x+w, 3))

        print(f"right diff: {diff2x, diff2y}")
        print(f"left diff: {diff1x, diff1y}")

        if diff2y == shiftup2+(rightSize[0]-shiftdown2) and diff1y == shiftup1+(leftSize[0]-shiftdown1):
            diffshiftup = shiftup2-shiftup1

            if diffshiftup >= 0:
                pano[diffshiftup:leftSize[0]+diffshiftup, :diff1x+w//2] = left_warp[:, :diff1x+w//2]
                pano[:rightSize[0], diff1x+w//2:] = right_warp[:, w//2:]
            else:
                pano[:leftSize[0], :diff1x+w//2] = left_warp[:, :diff1x+w//2]
                # Problem dimension here
                pano[-diffshiftup:leftSize[0]-diffshiftup, diff1x+w//2:] = right_warp[:, w//2:]
                # maybe this
                # pano[-diffshiftup:leftSize[0]-diffshiftup, diff1x+w//2:] = right_warp[:, w//2:]
                
        else:
            print("Alignement problem")

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
        
        shiftup1 = np.argmax(left_warp[:,0].mean(axis=1) != 0)
        shiftdown1 = shiftup1 + np.argmax(left_warp[shiftup1:, 0].mean(axis=1) == 0)
        shiftup2 = np.argmax(right_warp[:,0].mean(axis=1) != 0)
        shiftdown2 = shiftup2 + np.argmax(right_warp[shiftup2:, 0].mean(axis=1) == 0)
        

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
                pano[-diffshiftup:leftSize[0]-diffshiftup, diff1x+w//2:] = right_warp[:, w//2:]

        else:
            print("Alignement problem")

        # print(f"UDIS time : {t1-t0}")
        # print(f"Placement calculation time : {t2-t1}")
        # print(f"If code time : {time.time()-t2}")
        # print(f"Warp time : {time.time()-t0}")

        return pano.astype(np.uint8)
    

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
        first_loop = True

        while True:
            images, images_bool = readMemory(batchMMF, batchFlagPosition, imageCount, batchDataPosition, imageSize, imageWidth, imageHeight)
            
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
                
                write_memory(processedMMF, processedFlagPosition, processedDataPosition, processedImageSize, cv2.flip(panorama, 0))
                del panorama


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
            # keypoints, descriptors = self.ORB_extraction(images)
            t1 = time.time()
            matches_info, confidences = self.compute_matches_and_confidences(descriptors, keypoints)
            t2 = time.time()
            best_pairs = self.find_top_pairs(confidences)
            partial_order = find_cycle_for_360_panorama(confidences, front_image_index, False)[:-1]
            Hs, order, inverted = self.compute_homographies_and_order(keypoints, matches_info, partial_order)
            t3 = time.time()
            # Ms, order, inverted = self.compute_affines_and_order(keypoints, matches_info, partial_order)

            print("time to extract keypoints:", t1-t)
            print("time to compute matches:", t2-t1)
            print("time to compute homographies:", t3-t2)

            # put everything in the queues
            self.order_queue.put(order)
            # self.homography_queue.put(Hs)
            # self.homography_queue.put(Ms)
            # self.direction_queue.put(inverted)

            # print(f"Time to compute the Parameters: {time.time()-t}")
            print(order)
            time.sleep(10)
            # print("New parameters given")
            
            if debug:
                return keypoints, Hs, order, inverted, best_pairs, matches_info, confidences, images
                # return keypoints, Ms, order, inverted, best_pairs, matches_info, confidences, images

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

        order = None

        while True:
            if not self.order_queue.empty():
                del order
                order = self.order_queue.get()
                # Hs = self.homography_queue.get()
                # inverted = self.direction_queue.get()
                order = np.hstack((order, order, order))
                # Hs = np.concatenate((Hs, Hs, Hs))
            elif order is None:
                time.sleep(0.4)
                continue
            
            with lock:
                images = self.shared_images

            t = time.time()
            if self.warp_type == "cylindrical":
                images = [self.cylindricalWarp(img) for img in images]
            
            
            subset1, subset2 = self.chooseSubsetsAndTransforms(num_pano_img, order, headANgle)
            pano = self.UDIS_pano(images, subset1, subset2)
            # pano = self.UDIS_batch_pano(images, subset1, subset2)
            
            self.panoram_queue.put(pano)

            print(f"Time to make the Panorama: {time.time()-t}")
            # print("Panorama Given")
            if debug:
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
            boolean_list = [bool(b) for b in batchMMF.read(imageCount)]
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

    images = [cv2.resize(image, (1000,750)) for image in images]

    stitcher = custom_stitcher(camera_matrix=cam_mat, warp_type="cylindrica", algorithm=1, 
                                  trees=5, checks=50, ratio_thresh=0.9, score_threshold=0.1, device="cuda")
    
    stitcher.shared_images = images

    print("Second thread")
    _, _, _, _, _, _, _, images = stitcher.second_thread(front_image_index=0, debug=True)
    print("Third thread")
    stitcher.third_thread(headANgle = 70, num_pano_img=3, debug=True)
    print("FInished stitching")

    # for image in images:
    #     cv2.imshow("images after second thread", image)
    #     cv2.waitKey(2000)
    #     cv2.destroyAllWindows()

    panorama = stitcher.panoram_queue.get()
    panorama_resized = cv2.resize(panorama, dsize=None, fx=0.3, fy=0.3)
    cv2.imshow("panorama", panorama_resized)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    cv2.imwrite("panorama_UDIS.png", panorama)

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

    stitcher = custom_stitcher(camera_matrix=cam_mat, warp_type="cylindrical", algorithm=1, 
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

    stitcher = custom_stitcher(camera_matrix=cam_mat, warp_type="cylindrica", algorithm=1, trees=5, checks=50, ratio_thresh=0.7, score_threshold=0.0, device="cuda")

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

    stitcher = custom_stitcher(camera_matrix=cam_mat, warp_type="cylindrica", algorithm=1, trees=5, checks=50, ratio_thresh=0.7, score_threshold=0.05, device="cuda")

    first_thread = threading.Thread(target=stitcher.first_thread, args=(batchFlagPosition, imageCount, batchDataPosition, imageWidth, imageHeight,
                    processedFlagPosition, processedDataPosition, processedImageWidth, processedImageHeight, False))
    first_thread.daemon = True
    first_thread.start()

    sec_thread = threading.Thread(target=stitcher.second_thread, args=(0, False))
    sec_thread.daemon = True
    sec_thread.start()

    third_thread = threading.Thread(target=stitcher.third_thread, args=(0, 3, False))
    third_thread.daemon = True
    third_thread.start()

    print("Thread started")

    while True:
        time.sleep(100)
    

if __name__ == '__main__':
    
    # main function, the final result
    # main()

    # Test reading the images from shared memory and write an image in the panorama memory
    # test_reading_writing()

    # Test stitcher with own images
    test_stitcher()
    # test_stitcher()

    # Test threading function
    # test_threading()

    # Test one stitched camera
    # front_image_index = 0
    # angle = 0
    # num_pano_img = 3

    # test_one_image(front_image_index, angle, num_pano_img)