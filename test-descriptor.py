import argparse
import numpy as np

import torch
from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from lib.model_test import D2Net
from lib.utils import preprocess_image, preprocess_multipoint
from lib.pyramid import process_multiscale
# Added
from lib.utils import interpolate_dense_features

# Added
from multipoint.datasets import ImagePairDataset
import cv2
import matplotlib.pyplot as plt
import random
import yaml
from torch.utils.data import DataLoader
import os
import datetime

def main():

    # ---- Args ---- #
    parser = argparse.ArgumentParser(description='Project Test Script')
    # Important & required 
    parser.add_argument('-y', '--yaml-config', default='config/config_test_features.yaml', help='YAML config file')
    parser.add_argument('-o', '--output_folder', default=None, help='Output folder (after prefix). By default, set to results-<timestamp>')
    parser.add_argument('-p', '--plot', dest='plot', action='store_true', default=False, help='')
    # D2-Net 
    #parser.add_argument('--model_file', type=str, default='models/d2_tf.pth', help='path to the full model')
    parser.add_argument('--preprocessing', type=str, default=None, help='image preprocessing \'torch\' or None')
    parser.add_argument('--no-relu', dest='use_relu', default = True, action='store_false', help='remove ReLU after the dense feature extraction module') # Calling flag will store false
    parser.add_argument('--multiscale', dest='multiscale', action='store_true', default=False, help='extract multiscale features')
    # D2-net ones (that I won't need to change)
    parser.add_argument('--max_edge', type=int, default=1600, help='maximum image size at network input')
    parser.add_argument('--max_sum_edges', type=int, default=2800, help='maximum sum of image sizes at network input')
    # Other
    parser.add_argument('-s', '--seed', default=0, type=int, help='Seed of the random generators')

    args = parser.parse_args()
    if args.preprocessing == 'None': args.preprocessing = None

    # ---- YAML Config ---- #
    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # ---- Random Seed ---- #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #------------------------------------------------------

    # ---- Dataset ---- #
    dataset = ImagePairDataset(config['dataset'])  
    # TODO: Add these parameters to config file
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # ---- Device ---- #
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Device: " + str(device))

    # ---- Feature ---- #
    # TODO: MOVE THE FEATURE INSTANTIATION HERE******

    # ---- Model ---- #
    # Creating CNN model
    if 'd2' in config['feature']['type']:
        model = D2Net(
            model_file=config['feature']['model'],
            use_relu=args.use_relu,
            use_cuda=use_cuda
        )

    # ---- Matcher ---- #
    # Initialize matcher
    matcher = Matcher(config['matching'])

    # ---- Init Results ---- #
    results = {
        'points_repeated': 0,
        'points_optical': 0,
        'points_thermal': 0,
        'matches_correct': 0,
        'matches_total': 0,
        'iterations': 0,
    }

    #------------------------------------------------------
    # Load Images
    for idx, img_pair in tqdm(enumerate(dataloader), total=len(dataloader)):

    # ---- Use When Debugging: 
    #for idx in np.arange(2): 
    #    img_pair = dataset[idx] # keys: 'image', 'valid_mask', 'is_optical', 'name'
    # -------- End Debug

        # ---- Preprocessing PART 1 ---- #
        # Convert Image from [1,H,W], Range 0-1, Float Tensor to (H,W,3), 0-255, UINT8 np.ndarray
        img_optical = preprocess_multipoint(img_pair['optical']['image'])
        img_thermal = preprocess_multipoint(img_pair['thermal']['image'])

        # Set homographies
        if 'homography' not in img_pair['optical'].keys(): img_pair['optical']['homography'] = torch.eye(3).view(1,3,3) 
        if 'homography' not in img_pair['thermal'].keys(): img_pair['thermal']['homography'] = torch.eye(3).view(1,3,3) 

        # ---- Detect & Describe ---- #
        if config['feature']['type'] == 'd2-net':
            # D2-Net
            kp_optical, des_optical =  d2_net_detect_describe(args, img_optical, model, device)
            kp_thermal, des_thermal =  d2_net_detect_describe(args, img_thermal, model, device)

        elif config['feature']['type'] == 'sift':
            # SIFT
            # TODO: move the feature instantiation before the loop
            # TODO: Have 'nfeatures' as a kwargs parameter?
            feature = cv2.SIFT_create(nfeatures = 500)
            kp_optical, des_optical = feature.detectAndCompute(img_optical, None)
            kp_thermal, des_thermal = feature.detectAndCompute(img_thermal, None)

        elif config['feature']['type'] == 'd2-sift':
            # Initiate SIFT detector
            feature = cv2.SIFT_create(nfeatures = 500)

            # Keypoints
            kp_optical = feature.detect(img_optical, None)
            kp_thermal = feature.detect(img_thermal, None)

            # Describe & handle no kp
            if (len(kp_optical)>0): 
                kp_optical, des_optical = d2_net_describe(args, img_optical, kp_optical, model, device)
            else:
                des_optical = []

            if (len(kp_thermal)>0): 
                kp_thermal, des_thermal = d2_net_describe(args, img_thermal, kp_thermal, model, device)
            else:
                des_thermal = []

        elif config['feature']['type'] == 'd2-orb':
            # Initiate ORB detector
            feature = cv2.ORB_create()

            # Keypoints
            kp_optical = feature.detect(img_optical, None)
            kp_thermal = feature.detect(img_thermal, None)

            # Describe & handle no kp
            if (len(kp_optical)>0): 
                kp_optical, des_optical = d2_net_describe(args, img_optical, kp_optical, model, device)
            else:
                des_optical = []

            if (len(kp_thermal)>0): 
                kp_thermal, des_thermal = d2_net_describe(args, img_thermal, kp_thermal, model, device)
            else:
                des_thermal = []
                

        elif config['feature']['type'] == 'orb':
            # Initiate ORB detector
            feature = cv2.ORB_create()
            # Keypoints
            kp_optical, des_optical = feature.detectAndCompute(img_optical, None)
            kp_thermal, des_thermal = feature.detectAndCompute(img_thermal, None)

        else: 
            raise ValueError('Unsupported feature type. Supported options are d2-net and sift.')
            
        # ---- Mask ---- #
        kp_optical, des_optical = mask_keypoints(kp_optical, des_optical, img_pair['optical']['valid_mask'])
        kp_thermal, des_thermal = mask_keypoints(kp_thermal, des_thermal, img_pair['thermal']['valid_mask'])

        # ---- Match ---- #
        matches = matcher.match(des_optical, des_thermal)

        # ---- Outlier Rejection ---- #
        if config['matching']['ransac']:
            matches = ransac(kp_optical, kp_thermal, matches)

        # ---- Compute Repeatability ---- #
        # Compute
        repeatability, points_repeated, points_optical, points_thermal = compute_repeatability(kp_optical, img_pair['optical']['homography'], kp_thermal, img_pair['thermal']['homography'], threshold = config['eval']['repeat_thresh'])

        # Add to total stats
        results['points_repeated'] += points_repeated
        results['points_optical']  += points_optical
        results['points_thermal']  += points_thermal


        # ---- Compute MMA ---- #
        # Compute
        mma, m_correct, m_total = compute_mma(kp_optical, img_pair['optical']['homography'], kp_thermal, img_pair['thermal']['homography'], matches, threshold = config['eval']['match_thresh'])
        # Add to total stats
        results['matches_correct'] += m_correct
        results['matches_total']   += m_total
        results['iterations'] += 1

        #------------------------------------------------------

        # ---- Plot ---- #
        if args.plot and (len(kp_optical)>0) and (len(kp_thermal)>0):

            # Mask for viewing
            mask1 = img_pair['optical']['valid_mask'].numpy().squeeze()[:,:,np.newaxis]
            mask2 = img_pair['thermal']['valid_mask'].numpy().squeeze()[:,:,np.newaxis]

            # Draw Keypoints
            im_show('Keypoints (Optical, Thermal)', np.hstack((cv2.drawKeypoints(img_optical*mask1, kp_optical, None), cv2.drawKeypoints(img_thermal*mask2, kp_thermal, None))))

            # Draw matches
            im_show('Matches', cv2.drawMatches(img_optical*mask1, kp_optical, img_thermal*mask2, kp_thermal, matches, None, flags=cv2.DrawMatchesFlags_DEFAULT))


    # ---- Average Stats ---- #
    # Compute totall mma and repeatability
    results['repeatability'] = float(results['points_repeated']/(results['points_optical'] + results['points_thermal']))
    results['mma'] = float(results['matches_correct'] / results['matches_total'])
    # Average rest of stats
    results['avg_points_repeated']= float(results['points_repeated'] / results['iterations'])
    results['avg_points_optical'] = float(results['points_optical']  / results['iterations'])
    results['avg_points_thermal'] = float(results['points_thermal']  / results['iterations'])
    results['avg_matches_correct']= float(results['matches_correct'] / results['iterations'])
    results['avg_matches_total']  = float(results['matches_total']   / results['iterations'])
    # Nicely format
    results['points_repeated'] = int(results['points_repeated'])
    results['points_optical']  = int(results['points_optical']) 
    results['points_thermal']  = int(results['points_thermal']) 
    results['matches_correct'] = int(results['matches_correct']) 
    results['matches_total']   = int(results['matches_total']) 

    # ---- Save Results ---- #
    # Make directory
    if args.output_folder is not None: 
        results_folder = os.path.join(
            os.getcwd(), 
            config['saving']['results_prefix'], 
            args.output_folder) 
    else: 
        results_folder = os.path.join(
            os.getcwd(), 
            config['saving']['results_prefix'], 
            "results-{date:%Y-%m-%d-%H-%M-%S}".format(date=datetime.datetime.now()) ) 
    os.mkdir(results_folder)

    # Dump config, command line args, results
    with open(os.path.join(results_folder, 'params.yaml'), 'wt') as fh:
        yaml.dump({'args': args}, fh)
        yaml.dump({'yaml-config':config}, fh)

    with open(os.path.join(results_folder, 'results.yaml'), 'wt') as fh:
        yaml.dump(results, fh)

    return

class Matcher():
    def __init__(self, config):

        # Get parameters
        self.method = config['method']

        # Intiialize matcher
        # Option 1: Ratio Test
        if self.method == 'ratio':
            # Get Ratio
            try: 
                self.ratio = config['ratio']
            except: 
                raise ValueError('Ratio must be specified to use ratio method')
            
            # Initialize matcher
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        # Option 2: Crosscheck
        elif self.method=='crosscheck': 

            # Initialize matcher
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        else:
            raise ValueError('Unknown method. Only ratio and crosscheck accepted.')

    def match(self, des1, des2):

        # Handle one or both descriptors empty
        if (len(des1)==0) or (len(des2)==0): return []

        if self.method=='ratio': 
            # Handle too few descriptors
            if (len(des1)==1) or (len(des2)==1): return []

            knn_matches = self.matcher.knnMatch(des1, des2, k=2)
            # Ratio Test
            matches = []
            for first, second in knn_matches: 
                if second is None: continue
                if first.distance < self.ratio*second.distance: matches.append(first)

        elif self.method=='crosscheck': 
            matches = self.matcher.match(des1, des2)
            
        else: 
            raise ValueError('Unknown method. Only ratio and crosscheck accepted.')

        return matches

# Input: cv2.KeyPoints (u,v) format, matches
# Output: Inlier matches
def ransac(kp1, kp2, matches):

    # CASE 1: Insufficient matches (or no matches)
    if len(matches) < 4: return matches

    # CASE 2: Enough matches
    # Order points
    kp1_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    kp2_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # Apply ransac w/ homography model
    H, ransac_mask = cv2.findHomography(kp1_pts, kp2_pts, method=cv2.RANSAC) 

    # CASE 3: Ransac failed
    if H is None:
        return matches

    # Keep only the inliers
    inlier_matches = tuple(matches[i] for i in ransac_mask[:,0].nonzero()[0]) 

    return inlier_matches

# Takes in cv2.KeyPoints (u,v format), the homographies, matches, and threshold
# Outputs mma (correct/total), # correct, and total # of matches
def compute_mma(kp_source, H_source, kp_dest, H_dest, matches, threshold):

    # ERROR CASE: (i) One of the keypoints is empty
    if (len(kp_source)==0) or (len(kp_dest)==0): return 0.0, 0, 0
    # ERROR CASE: (ii) Matches empty
    if (len(matches) == 0): return 0.0, 0, 0

    # Order keypoints based on matches
    kp_s_ordered = [kp_source[m.queryIdx] for m in matches]
    kp_d_ordered = [kp_dest[m.trainIdx]   for m in matches]

    # Convert cv2 Keypoints (u,v format) to [N,2] (x,y) format
    kp_s = cv2_to_xy(kp_s_ordered)
    kp_d = cv2_to_xy(kp_d_ordered)
    
    # Rectify keypoints
    kp_s_rect = warp_xy_to_xy(kp_s, np.linalg.inv(H_source))
    kp_d_rect = warp_xy_to_xy(kp_d, np.linalg.inv(H_dest))

    # Get # of correct matches
    m_correct = get_correct_matches(kp_s_rect, kp_d_rect, threshold)

    # Finalize & return
    m_total = len(matches)
    mma = m_correct / m_total

    return mma, m_correct, m_total

# Takes in keypoints of form [M,2], (x,y) format, (M = # matches)
# Outputs the number of matches that are correct
# Assumes they are already rectified & ordered
def get_correct_matches(kp_s_rect, kp_d_rect, threshold):

    # Get distances between each matc
    distances = np.linalg.norm(kp_s_rect - kp_d_rect, axis=1)  

    # Determine how many are within threshold
    m_correct = np.sum(distances < threshold).astype(int)

    return m_correct

# Takes in cv2 Keypoints of form (u,v)
# Output: Total # of repeated keypoints across *both* images
# (need to divide by 2 to get average number)
def compute_repeatability(kp1, H1, kp2, H2, threshold):
    # CASE: One of keypoints is empty
    if (len(kp1)==0) or (len(kp2)==0): return 0.0, 0, len(kp1), len(kp2)

    # Convert cv2 Keypoints (u,v format) to [N,2] (x,y) format
    kp1 = cv2_to_xy(kp1)
    kp2 = cv2_to_xy(kp2)
    
    # Rectify keypoints
    kp1_rect = warp_xy_to_xy(kp1, np.linalg.inv(H1))
    kp2_rect = warp_xy_to_xy(kp2, np.linalg.inv(H2))

    # Get repeated points for each
    # kp1_rep and kp2_rep are not necessarily the same. 
    kp1_rep, kp2_rep = get_repeated_points(kp1_rect, kp2_rect, threshold)

    # Finalize & return
    kp1_tot = kp1.shape[0]
    kp2_tot = kp2.shape[0]
    repeatability = (kp1_rep + kp2_rep)/(kp1_tot + kp2_tot)

    return repeatability, (kp1_rep+kp2_rep), kp1_tot, kp2_tot

# Assumes rectified images (or at least warped by same homography)
def get_repeated_points(kp1, kp2, threshold):

    # distances.shape = [len(kp1), len(kp2)]
    distances = np.linalg.norm(kp1[:,np.newaxis,:] - kp2, axis=2)

    # For *each* keypoint in kp1 and kp2, get distance to *closest* point 
    kp1_dist = np.min(distances, axis=1)
    kp2_dist = np.min(distances, axis=0)

    # Get number of points where closest point under threshold
    kp1_rep = np.sum(kp1_dist < threshold).astype(int)
    kp2_rep = np.sum(kp2_dist < threshold).astype(int)

    return kp1_rep, kp2_rep

# Input:  [N,2] array of keypoints (x,y form)
# Output: [N,2] array of keypoints (x,y form)
def warp_xy_to_xy(kp, H):

    # Convert to homogenous format: [3,N], (u,v) format
    kp_aug = np.vstack((kp[:,1], kp[:,0], torch.ones((kp.shape[0]))))

    # Apply homography
    if len(H.shape) == 3: H = H.squeeze()
    kp_warped_aug = H @ kp_aug

    # Reproject (divide by last element), convert back to (x,y) from (u,v)
    # Output shape: [N,2]
    kp_warped = (np.vstack((kp_warped_aug[1,:], kp_warped_aug[0,:])) / kp_warped_aug[2,:]).T

    return kp_warped

# Input: list of cv2.KeyPoints (already in u,v format)
# Output: [N,2] array of keypoints (x,y format)
def cv2_to_xy(kp_list):
    kp_xy = np.float32([[kp.pt[1], kp.pt[0]] for kp in kp_list]).reshape(-1,2)
    return kp_xy

# TODO: Could update to look at all 4 pixels around the keypoint. Right now just rounding to closest pixel
def mask_keypoints(keypoints, descriptors, valid_mask_tensor):

    # Handle no keypoints or descriptors
    if (len(keypoints)==0) or (len(descriptors)==0): return [], []

    # Create a mask by checking if it's in the valid region
    kp_mask = [int(valid_mask_tensor.squeeze()[round(kp.pt[1]), round(kp.pt[0])].item()) for kp in keypoints]

    # Retain only valid keypoints & descriptors
    #kp_valid  = keypoints[kp_mask]
    kp_valid  = tuple(kp for i, kp in enumerate(keypoints) if kp_mask[i])
    des_valid = descriptors[np.array(kp_mask).nonzero()[0], :]

    #valid_keypoints = [kp for kp in keypoints if valid_mask_tensor.squeeze()[round(kp.pt[1]), round(kp.pt[0])]]
    
    return kp_valid, des_valid

# TODO: update to include preprocessing?
def im_show(str,img):
    cv2.imshow(str,img)
    cv2.waitKey()
    cv2.destroyWindow(str)   

def d2_net_detect_describe(args, image, model, device):

    # Resize image if too large
    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    if max(resized_image.shape) > args.max_edge:
        resized_image = scipy.misc.imresize(
            resized_image,
            args.max_edge / max(resized_image.shape)
        ).astype('float')
    if sum(resized_image.shape[: 2]) > args.max_sum_edges:
        resized_image = scipy.misc.imresize(
            resized_image,
            args.max_sum_edges / sum(resized_image.shape[: 2])
        ).astype('float')

    # Get scale factor for resizing
    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    # ---- Preprocessing PART 2 ---- #
    input_image = preprocess_image(
        resized_image,
        preprocessing=args.preprocessing
    )

    # Added by Mia
    model.eval()

    # Get keypoints, scores, descriptors
    with torch.no_grad():
        if args.multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales=[1]
            )

    # Input image coordinates
    # Upscale locations by reduction factor from earlier
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j

    # Convert to image coordinates
    # i, j -> u, v
    # Rearrange the columns from [a, b, c] to [b, a, c]
    # TODO: Figure out if you should remove this
    #keypoints = keypoints[:, [1, 0, 2]] # Note, combined this step with converting to cv2 keypoints below

    # Convert Keypoints to cv2 Keypoints
    # TODO: add radius to config file
    RADIUS = 4
    cv_keypoints = [cv2.KeyPoint(point[1], point[0], RADIUS, response=scores[i]) for i,point in enumerate(keypoints)]

    return cv_keypoints, descriptors

def d2_net_describe(args, image, kp, model, device):

    # Handle keypoints
    keypoints = cv2_to_xy(kp)    

    # ---- Preprocessing PART 2 ---- #
    input_image = preprocess_image(
        image,
        preprocessing=args.preprocessing
    )

    # Added by Mia
    model.eval()

    # Get keypoints, scores, descriptors
    with torch.no_grad():

        # Get dense features
        dense_features = model.dense_feature_extraction(torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32),device=device))

        _, _, h, w = dense_features.size()
        
        # Scale Keypoints
        kp_i_scale = input_image.shape[1]/h
        kp_j_scale = input_image.shape[2]/w
        keypoints[:,0] /= kp_i_scale
        keypoints[:,1] /= kp_j_scale

        # Get Descriptor Value
        raw_descriptors, pos, ids = interpolate_dense_features(torch.tensor(keypoints.T).to(device), dense_features[0])
        
        # Turn descriptors into list
        kp_new = [kp[id] for id in ids]
        descriptors = raw_descriptors.cpu().numpy().T

    return kp_new, descriptors


'''
# ---- Plot ---- #
fig, ax = plt.subplots(1,2)
# Plot optical
ax[0].set_title('Optical Keypoints')
ax[0].imshow(img_optical)
ax[0].scatter(kp_optical[:,0], kp_optical[:,1], c='red', marker='o', s=0.75)
ax[0].axis('off')
# Plot thermal
ax[1].set_title('Thermal Keypoints')
ax[1].imshow(img_thermal)
ax[1].scatter(kp_thermal[:,0], kp_thermal[:,1], c='red', marker='o', s=0.75)
ax[1].axis('off')
plt.show()
'''

'''
# Save
if args.output_type == 'npz':
    with open(path + args.output_extension, 'wb') as output_file:
        np.savez(
            output_file,
            keypoints=keypoints,
            scores=scores,
            descriptors=descriptors
        )
elif args.output_type == 'mat':
    with open(path + args.output_extension, 'wb') as output_file:
        scipy.io.savemat(
            output_file,
            {
                'keypoints': keypoints,
                'scores': scores,
                'descriptors': descriptors
            }
        )
else:
    raise ValueError('Unknown output type.')
'''

if __name__ == "__main__":
    main()
