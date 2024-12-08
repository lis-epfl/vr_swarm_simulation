# import custom_stitching
import numpy as np
import cv2
import torch
import time
from transformers import SuperPointForKeypointDetection
# from torch.quantization import quantize_dynamic
from numba import jit
import networkx as nx
import random


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

class BaseStitcher:
    def __init__(self, 
                 camera_matrix = None, 
                 cylindricalWarp=False, 
                 full_cylinder = False, 
                 algorithm=1, 
                 trees=5, 
                 checks=50, 
                 ratio_thresh = 0.7, 
                 score_threshold = 0.2,
                 active_matcher_type= "BF",
                 isRANSAC= False, 
                 device = "cpu"
                 ):
        
        self.superpoint_model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
        # self.model = quantize_dynamic(
        #     self.model,  # the original model
        #     {torch.nn.Conv2d},  # layers to quantize
        #     dtype=torch.qint8  # quantization data type
        #     )
        self.superpoint_model.eval()
        self.device = torch.device(device)
        
        self.W_SP = 640
        self.H_SP = 480
        self.score_threshold = score_threshold
        self.superpoint_model.to(self.device)

        # Flann parameters. Algorithm that match the different keypoints between images based on the descriptors
        self.checks = checks
        self.index_params = dict(algorithm=algorithm, trees=trees)
        self.search_params = dict(checks=checks) 

        # Create a FLANN and BF Matcher
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        self.BF = cv2.BFMatcher()
        self.ratio_thresh = ratio_thresh
        self.active_matcher_type = active_matcher_type
        self.isRANSAC = isRANSAC

        # Camera informations
        if camera_matrix is None:
            camera_matrix = np.array([[300,0, 150], [0,300, 150], [0,0, 1]])
        self.camera_matrix = camera_matrix
        self.focal = camera_matrix[0,0]

        # Warp informations
        self.cylindricalWarp = cylindricalWarp
        #self.full_cylinder = full_cylinder # not implemeneted (for full 360 degrees stitching)

        # T keep in memory the "remap matrix"
        self.points_remap = None

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

    def CylindricalWarp(self, img):

        """ 
        taken from: https://github.com/saurabhkemekar/Image-Mosaicing/blob/master/cylinder_stiching.py

        Warps an image in cylindrical coordinate based on the intrinsic camera matrix.
        """
        if self.points_remap is None:
            K = self.camera_matrix.copy()
            K[0,2] = img.shape[1]//2
            K[1,2] = img.shape[0]//2
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
            - images : When cylindricalWarp is True then it outputs the cylindrical images
        
        """""
        if self.cylindricalWarp:
            images_cyl = [self.CylindricalWarp(img) for img in images]
            if images_cyl ==[]:
                return None, None

            H, W = images_cyl[0].shape[:2]
            if W<self.W_SP or H<self.H_SP:
                inputs = [cv2.resize(img, (self.W_SP, self.H_SP), interpolation=cv2.INTER_LINEAR) for img in images_cyl]
            else:
                inputs = [cv2.resize(img, (self.W_SP, self.H_SP), interpolation=cv2.INTER_AREA) for img in images_cyl]
        else:
            H, W = images[0].shape[:2]
            if W<self.W_SP or H<self.H_SP:
                inputs = [cv2.resize(img, (self.W_SP, self.H_SP), interpolation=cv2.INTER_LINEAR) for img in images]
            else:   
                inputs = [cv2.resize(img, (self.W_SP, self.H_SP), interpolation=cv2.INTER_AREA) for img in images]
            
        
        rescale_factor = 0.003921568859368563
        inputs = torch.FloatTensor(np.array(inputs)).permute(0,3,1,2).to(self.superpoint_model.device)*rescale_factor
        with torch.no_grad():
            outputs = self.superpoint_model(inputs)
        
        keypoints, scores, descriptors = outputs['keypoints'].cpu().numpy(), outputs['scores'].cpu().numpy(), outputs['descriptors'].cpu().numpy()

        ratio_y, ratio_x = H/self.H_SP, W/self.W_SP

        if self.cylindricalWarp:
            return {'keypoints': keypoints, 'scores': scores, 'descriptors': descriptors}, (ratio_x, ratio_y), images_cyl

        images = [np.array(image) for image in images]
        return {'keypoints': keypoints, 'scores': scores, 'descriptors': descriptors}, (ratio_x, ratio_y)

    def keep_best_keypoints(self, outputs, ratios, image_height):
        # Extract tensors from the outputs dictionary
        kpts, scores, dpts = outputs['keypoints'], outputs['scores'],  outputs['descriptors']

        # Get mask of valid scores
        score_mask = scores > self.score_threshold
        threshold = 2 * image_height / 3
        
        # Keep keypoints with best score and within the above 2/3 of the image
        valid_mask = score_mask | (kpts[:,:, 1]<threshold)

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
                
                if self.active_matcher_type == "BF":
                    matches = self.BF_matching(desc1, desc2)
                else:
                    ## Need to change FLANN parameters if we want to fine tune 
                    matches = self.FLANN_matching(desc1, desc2)
                
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

    def chooseSubsetsAndTransforms(self, Ts, num_pano_img, order, headAngle):
        num_images = len(Ts)//3
        headAngle -= headAngle//360 *360
        if headAngle<0:
            headAngle+=360
        angle_per_image, angle_rad= 2*np.pi/num_images, np.deg2rad(headAngle)#- 2*np.pi(angle>180)
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
                if self.isRANSAC:
                    Hs[i], _ = cv2.findHomography(dst_p, src_p, method=cv2.RANSAC, ransacReprojThreshold=3, confidence=0.995)
                else:
                    Hs[i], _ = cv2.findHomography(dst_p, src_p, method=0)

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
        pass
        
        # Initial dimensions of the first image
        # h, w = images[0].shape[:2]

        # # Initial corners of the reference image
        # corners = np.array([[0, w-1 , w -1, 0],
        #                       [0, 0, h-1 , h-1 ],
        #                       [1, 1, 1, 1]], dtype=np.float32)

        # if inverted:
        #     Ms2 = invert_affine_matrices(Ms2)
        # else:
        #     Ms1 = invert_affine_matrices(Ms1)

        # # First, apply affine transformations for subset1 (left side)
        # warped_corners_1, M1_acc = apply_affine_matrices(Ms1, corners)

        # # Then, apply affine transformations for subset2 (right side)
        # warped_corners_2, M2_acc = apply_affine_matrices(Ms2, corners)

        # # Calculate the bounding box for the entire panorama
        # all_corners = np.concatenate((warped_corners_1, warped_corners_2), axis=1)

        # x_min, x_max = np.int32(all_corners[0, :].min()), np.int32(all_corners[0, :].max())
        # y_min, y_max = np.int32(all_corners[1, :].min()), np.int32(all_corners[1, :].max())

        # # print(x_min, x_max, y_min, y_max)

        # x_min, x_max = max(x_min, -clip_x), min(x_max, clip_x)
        # y_min, y_max = max(y_min, -clip_y), min(y_max, clip_y)

        # # print(x_min, x_max, y_min, y_max)

        # # Translation matrix for panorama placement
        # translation_matrix = np.array([[1, 0, -x_min],
        #                             [0, 1, -y_min]], dtype=np.float32)

        # panorama_size = (x_max - x_min, y_max - y_min)

        # # Warp the reference image and place it on the panorama canvas using cv2.warpAffine
        # panorama = cv2.warpAffine(images[subset2[0]], translation_matrix, panorama_size)
        # M1_acc[:,:2,-1]+=translation_matrix[:2,-1]
        # M2_acc[:,:2,-1]+=translation_matrix[:2,-1]

        # # Warp and blend images from subset1 (left side), skipping the reference image
        # for i in range(M1_acc.shape[0]):
        #     M_translate = M1_acc[i]

        #     warped_img = cv2.warpAffine(images[subset1[i + 1]], M_translate, panorama_size)

        #     mask = (warped_img > 0).astype(np.uint8)
        #     panorama[mask > 0] = warped_img[mask > 0]

        # # Warp and blend images from subset2 (right side), skipping the reference image
        # for i in range(M2_acc.shape[0]):
        #     M_translate = M2_acc[i]

        #     warped_img = cv2.warpAffine(images[subset2[i + 1]], M_translate, panorama_size)

        #     mask = (warped_img > 0).astype(np.uint8)
        #     panorama[mask > 0] = warped_img[mask > 0]

        # return panorama
    
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

    def findHomographyOrder(self, images, front_image_index=0,Hs=None, verbose = False, debug= False):
        """""
        This method uses some of the above methods to extract the order and the homographies of the paired images.
        Input:
            - images : list of NDArrays.
            - front_image_index : the index of the front image of the pilot
            - Hs : previously computed homographies
            - verbose : if True, print the times
            - debug : if True, return every element previously computed
        """""

        t0 = time.time()
        if self.cylindricalWarp:
            outputs, ratios, images = self.SP_inference_fast(images)
        else:
            outputs, ratios = self.SP_inference_fast(images)

        if outputs is None:
            return None, None, None
        
        keypoints, descriptors = self.keep_best_keypoints(outputs, ratios, images[0].shape[0])
        t1 = time.time()
        matches_info, confidences = self.compute_matches_and_confidences(descriptors, keypoints)
        t2 = time.time()
        partial_order = find_cycle_for_360_panorama(confidences, front_image_index, False)[:-1]

        H, order, inverted = self.compute_homographies_and_order(keypoints, matches_info, partial_order, Hs)
        t3 = time.time()

        if Hs is None or H.shape !=Hs.shape:
            if Hs is not None:
                print(H.shape, Hs.shape)
            Hs = np.zeros_like(H)
        
        h, w = images[0].shape[:2]

        # Initial corners of the reference image
        # corners = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]], dtype=np.float32).reshape(-1, 1, 2)
        corners = np.array([[0, w-1 , w -1, 0],
                              [0, 0, h-1 , h-1 ],
                              [1, 1, 1, 1]], dtype=np.float32)
        
        Hs = ControlHomography(H, Hs, corners, ratio=1.5, change_thresh =200)
        
        t4 = time.time()

        if verbose:
            print("time to extract keypoints:", t1-t0)
            print("time to compute matches:", t2-t1)
            print("time to compute homographies:", t3-t2)
            print("Time to control:", t4-t3)
            print("Total time to compute Hs and order:", t4-t0)
        
        if debug:
            best_pairs = self.find_top_pairs(confidences)
            return keypoints, Hs, order, inverted, best_pairs, matches_info, confidences
        
        return Hs, order, inverted

    def stitch(self, images, order, Hs, inverted, headAngle, processedImageWidth, processedImageHeight, num_pano_img=3, verbose=False):
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

        # order, Hs, inverted = None, None, None

        t = time.time()
        if self.cylindricalWarp:
            images = [self.CylindricalWarp(img) for img in images]

        subset1, subset2, Hs1, Hs2 = self.chooseSubsetsAndTransforms(Hs, num_pano_img, order, headAngle)
        print(subset1, subset2)
        pano = self.compose_with_defined_size(images, Hs1, Hs2, subset1, subset2, inverted, panoWidth=processedImageWidth, panoHeight=processedImageHeight)            
        if verbose:
            print(f"Warp time: {time.time()-t}")    
        return pano
            

# Have to add function that will look if parameters have changed and if so, recompute cylindrical warping
# Think about how to change the stitcher. Maybe instead of 3 methods as thread, just do 3 functions

def hasSmallStretchHomography(H: np.ndarray, corners, ratio = 1.5):

    H[:2, -1] = np.zeros(2)
    _, newCorners =apply_homographies(np.expand_dims(H, axis=0), corners)

    if newCorners>1.5*corners[:2]:
        print("Too big changes")
        return False, None
    
    return True

def hassmallChangeHomography(H_new, H_prev, criterion = 0.5):
    """
    Assume same dimensions for H_new and H_prev
    """
    return np.linalg.norm(H_prev-H_new, 'fro')>criterion

# def ControlHomography(Hs_new, Hs_prev, corners, ratios=1.5, change_thresh =100):

#     b = Hs_new.shape[0]

#     # Compute new corners in batch
#     Hs_new_ = Hs_new.copy()
#     Hs_new_[:, :2, -1] = np.zeros(2)

#     new_corners_2d = (Hs_new_@corners[:, 1:3])[:, :2, :]

#     Hs_prev_ = Hs_prev.copy()
#     Hs_prev_[:, :2, -1] = np.zeros(2)
#     prev_corners_2d = (Hs_prev_@corners[:, 1:3])[:, :2, :]

#     good_index = np.ones(b, dtype=np.bool_)
#     # Control if image not too stretched
#     for i in range(b):
#         if np.linalg.norm(new_corners_2d[i]-corners[:2, 1:3], "fro")>ratios**2:
#             good_index[i]=False
#         else:
#             if Hs_prev[i] == np.zeros_like(Hs_prev[i]):
#                 Hs_prev[i] = Hs_new[i]
#                 good_index[i]=False

#     # Control if points have not changed too much between 2 homographies
#     for i in range(b):
#         distance = np.linalg.norm(new_corners_2d[i]-prev_corners_2d[i], "fro")
#         if good_index[i] and distance < change_thresh :
#             Hs_prev[i] = Hs_new[i]

#     return Hs_prev

def ControlHomography(Hs_new, Hs_prev, corners, ratio=2.5, change_thresh=200):
    """
    Validate and update homographies based on stretch and change thresholds.

    Parameters:
        Hs_new (np.ndarray): New homographies, shape (b, 3, 3).
        Hs_prev (np.ndarray): Previous homographies, shape (b, 3, 3).
        corners (np.ndarray): Corner points, shape (n, 3, num_corners).
        ratios: Maximum allowed ratio for homography stretch.
        change_thresh (float): Maximum allowed change between consecutive homography transformations on the two right corners.

    Returns:
        np.ndarray: Updated homographies, shape (b, 3, 3).
    """
    b = Hs_new.shape[0]

    # Remove translation component
    Hs_new_ = Hs_new.copy()
    Hs_new_[:, :2, -1] = 0

    Hs_prev_ = Hs_prev.copy()
    Hs_prev_[:, :2, -1] = 0

    # Compute transformed corners
    # new_corners_2d = np.einsum('bij,jk->bik', Hs_new_, corners[:, 1:3])[:, :2, :]
    # prev_corners_2d = np.einsum('bij,jk->bik', Hs_prev_, corners[:, 1:3])[:, :2, :]

    new_corners = np.einsum('bij,jk->bik', Hs_new_, corners[:, 1:3])#[:, :2, :]
    prev_corners = np.einsum('bij,jk->bik', Hs_prev_, corners[:, 1:3])#[:, :2, :]

    new_corners_2d = new_corners[:, :2, :] / (new_corners[:, -1:, :] + 1e-6)
    prev_corners_2d = prev_corners[:, :2, :] / (prev_corners[:, -1:, :] + 1e-6)

    good_index = np.ones(b, dtype=bool)

    # Check for image stretch
    for i in range(b):
        # frobenius_distance = np.linalg.norm(new_corners_2d[i] - corners[:2, 1:3], ord="fro")
        dist_new_corners = np.linalg.norm(new_corners_2d[i], axis=0)
        dist_corners = np.linalg.norm(corners[:2, 1:3], axis=0)

        # print(dist_new_corners)
        # print(ratio*dist_corners)
        # print(ratio*dist_corners < dist_new_corners)
        if np.allclose(Hs_prev[i], 0):  # Check for zero homography
            if np.any(ratio*dist_corners < dist_new_corners):
                # print("too large homography")
                continue
            Hs_prev[i] = Hs_new[i]
            good_index[i] = False
            # print("Zero homography but new one good")
    
    # Check if two changes are small enough
    for i in range(b):
        if good_index[i]:
            distance = np.linalg.norm(new_corners_2d[i] - prev_corners_2d[i], axis=0)
            if np.any(distance>change_thresh):
                # print("Changes are too big")
                continue
            Hs_prev[i] = Hs_new[i]
            # print(f"CHanges in homography {i}")

    return Hs_prev



    #Control if not too much changes between two consecutive homorgraphies

