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
from multipoint.datasets import ImagePairDataset
import cv2
import matplotlib.pyplot as plt
import random
import yaml


def main():

    # ---- Args ---- #
    parser = argparse.ArgumentParser(description='Project Test Script')
    # D2-Net 
    parser.add_argument('--model_file', type=str, default='models/d2_tf.pth', help='path to the full model')
    parser.add_argument('--multiscale', dest='multiscale', action='store_true', default=False, help='extract multiscale features')
    parser.add_argument('--no-relu', dest='use_relu', default = True, action='store_false', help='remove ReLU after the dense feature extraction module') # Calling flag will store false
    parser.add_argument('--preprocessing', type=str, default='torch', help='image preprocessing (caffe or torch)')
    parser.add_argument('--max_edge', type=int, default=1600, help='maximum image size at network input')
    parser.add_argument('--max_sum_edges', type=int, default=2800, help='maximum sum of image sizes at network input')
    # Output
    parser.add_argument('--output_extension', type=str, default='.d2-net', help='extension for the output')
    parser.add_argument('--output_type', type=str, default='npz', help='output file type (npz or mat)')
    # Added
    parser.add_argument('-s', '--seed', default=0, type=int, help='Seed of the random generators')
    parser.add_argument('-y', '--yaml-config', default='config/config_test_features.yaml', help='YAML config file')
    args = parser.parse_args()

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

    # ---- Device ---- #
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # ---- Model ---- #
    # Creating CNN model
    model = D2Net(
        model_file=args.model_file,
        use_relu=args.use_relu,
        use_cuda=use_cuda
    )

    # ---- Matcher ---- #
    # Initialize matcher
    matcher = cv2.BFMatcher()

    # ---- Init Results ---- #
    results = {
        'repeated_points': 0,
        'total_points': 0,
        'repeatability': 0,
        'correct_matches': 0,
        'total_matches': 0,
        'matching_accuracy': 0,
        'iterations': 0,
    }

    #------------------------------------------------------
    # TODO: INSERT LOOP HERE

    # Load Image
    img_pair = dataset[0] # keys: 'image', 'valid_mask', 'is_optical', 'name'

    # ---- Preprocessing PART 1 ---- #
    # Convert Image from [1,H,W], Range 0-1, Float Tensor to (H,W,3), 0-255, UINT8 np.ndarray
    img_optical = preprocess_multipoint(img_pair['optical']['image'])
    img_thermal = preprocess_multipoint(img_pair['thermal']['image'])

    # ---- Detect & Describe ---- #
    if config['feature_type'] == 'd2-net':
        # D2-Net
        kp_optical, des_optical =  d2_net_detect_describe(args, img_optical, model, device)
        kp_thermal, des_thermal =  d2_net_detect_describe(args, img_thermal, model, device)

    elif config['feature_type'] == 'sift':
        # SIFT
        feature = cv2.SIFT_create(nfeatures = 500)
        kp_optical, des_optical = feature.detectAndCompute(img_optical, None)
        kp_thermal, des_thermal = feature.detectAndCompute(img_thermal, None)
    
    else: 
        raise ValueError('Unsupported feature type. Supported options are d2-net and sift.')
        
    # Mask keypoints & descriptors to valid regions
    kp_optical, des_optical = mask_keypoints(kp_optical, des_optical, img_pair['optical']['valid_mask'])
    kp_thermal, des_thermal = mask_keypoints(kp_thermal, des_thermal, img_pair['thermal']['valid_mask'])

    # ---- Match ---- #
    # Get top two closest matches w/ bf matcher
    knn_matches = matcher.knnMatch(des_optical, des_thermal, k=2)
    # Ratio Test
    matches = []
    for first, second in knn_matches: 
        if first.distance < config['matching']['ratio']*second.distance: matches.append(first)

    # ---- Outlier Rejection ---- #
    # Order points
    kp_optical_pts = np.float32([kp_optical[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    kp_thermal_pts = np.float32([kp_thermal[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    # Apply ransac w/ homography model
    __, ransac_mask = cv2.findHomography(kp_optical_pts, kp_thermal_pts, method=cv2.RANSAC) 
    # Keep only the inliers
    matches = tuple(matches[i] for i in ransac_mask[:,0].nonzero()[0]) 

    # ---- Compute Repeatability ---- #
    # TODO implement
    '''
    repeatability, repeated_points, total_points = compute_repeatability(kp_optical, img_pair['optical']['homography'], kp_thermal, img_pair['thermal']['homography'], threshold = config['repeatability']['threshold'])
    '''
    # TODO add to total stats

    # ---- Compute MMA ---- #
    # TODO implement
    '''
    mma, correct_matches, total_matches = compute_correct_matches(kp_optical, img_pair['optical']['homography'], kp_thermal, img_pair['thermal']['homography'], matches, threshold = config['matching']['threshold'])
    '''
    # TODO add to total stats

    #------------------------------------------------------

    # TODO: Compute average stats


    # TODO: Save results


    # ---- Plot ---- #
    # Draw Keypoints
    im_show('Keypoints (Optical, Thermal)', np.hstack((cv2.drawKeypoints(img_optical, kp_optical, None), cv2.drawKeypoints(img_thermal, kp_thermal, None))))

    # Draw matches
    im_show('Matches', cv2.drawMatches(img_optical, kp_optical, img_thermal, kp_thermal, matches, None, flags=cv2.DrawMatchesFlags_DEFAULT))


    return

# TODO: May need to update (technically should check all 4 pixels around non-integer keypoint value)
def mask_keypoints(keypoints, descriptors, valid_mask_tensor):
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
    #keypoints = keypoints[:, [1, 0, 2]]

    # TODO: Convert Keypoints to cv2 Keypoints
    RADIUS = 4
    cv_keypoints = [cv2.KeyPoint(point[1], point[0], RADIUS, response=scores[i]) for i,point in enumerate(keypoints)]

    return cv_keypoints, descriptors



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