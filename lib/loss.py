import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn.functional as F

from lib.utils import (
    grid_positions,
    upscale_positions,
    downscale_positions,
    savefig,
    imshow_image
)
from lib.exceptions import NoGradientError, EmptyTensorError

matplotlib.use('Agg')

# added
import os

# New loss function 
# Instead of using depth, instrinics, poses, and bboxs
# Correspondences are found with homographies and masks
def loss_function(
        model, batch, device, margin=1, safe_radius=4, scaling_steps=3, plot=False, plot_path=None, weighted_loss=False
):
    output = model({
        'image1': batch['image1'].to(device),
        'image2': batch['image2'].to(device)
    })

    loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
    has_grad = False

    n_valid_samples = 0
    for idx_in_batch in range(batch['image1'].size(0)):

        '''
        # Annotations
        depth1 = batch['depth1'][idx_in_batch].to(device)  # [h1, w1]
        intrinsics1 = batch['intrinsics1'][idx_in_batch].to(device)  # [3, 3]
        pose1 = batch['pose1'][idx_in_batch].view(4, 4).to(device)  # [4, 4]
        bbox1 = batch['bbox1'][idx_in_batch].to(device)  # [2]

        depth2 = batch['depth2'][idx_in_batch].to(device)
        intrinsics2 = batch['intrinsics2'][idx_in_batch].to(device)
        pose2 = batch['pose2'][idx_in_batch].view(4, 4).to(device)
        bbox2 = batch['bbox2'][idx_in_batch].to(device)
        '''

        #-------------#
        # Annotations #
        #-------------#
        # Homographies
        if 'homography' in batch['optical'].keys(): 
            homography1 = batch['optical']['homography'][idx_in_batch].squeeze().float().to(device)
            homography2 = batch['thermal']['homography'][idx_in_batch].squeeze().float().to(device)
        else:
            homography1 = torch.tensor(np.eye(3)).view(3,3).float().to(device)
            homography2 = torch.tensor(np.eye(3)).view(3,3).float().to(device)
        # Masks
        mask1 = batch['optical']['valid_mask'][idx_in_batch].float().squeeze().to(device)
        mask2 = batch['thermal']['valid_mask'][idx_in_batch].float().squeeze().to(device)
        #----------------

        # Network output
        # dense_features1.shape = [512, 30, 40] = [N_channel, 240/8, 320/8]
        dense_features1 = output['dense_features1'][idx_in_batch]
        # c = 512, h1 = 30, w1 = 40
        c, h1, w1 = dense_features1.size()
        # output['scores1'][idx_in_batch].shape = [30, 40]
        # scores1.shape = [1200] = [30 x 40] => flattened so we can mask it w/ 'ids'
        scores1 = output['scores1'][idx_in_batch].view(-1)

        #------
        # TODO: In a future iteration, could try masking here as well to ensure
        # all invalid descriptors are excluded from the loss function. 
        # (Since 'all descriptors' are used in the negative_distance)
        # For now, I'm setting 'border_reflect' to false in the settings,
        # which blacks out the invalid regions
        #------

        # dense_features2.shape = [512, 30, 40] = [N_channel, 240/8, 320/8]
        dense_features2 = output['dense_features2'][idx_in_batch]
        # h2 = 30, w2 = 40
        _, h2, w2 = dense_features2.size()
        # scores2.shape = [30, 40] =/= scores1.shape
        scores2 = output['scores2'][idx_in_batch]

        # all_descriptors1.shape = [512, 1200] = [512, 30x40] (flattened so we can mask it w/ 'ids')
        all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
        # descriptors1.shape = [512, 1200] (flattened so we can mask it w/ 'ids')
        descriptors1 = all_descriptors1

        # all_descriptors2.shape = [512, 1200] = [512, 30x40] (flattened so we can mask it w/ 'ids')
        all_descriptors2 = F.normalize(dense_features2.view(c, -1), dim=0)

        # Warp the positions from image 1 to image 2
        # fmap_pos1.shape = [2, 1200] = [2, 30x40]
        # fmap_pos1 = [ 0 0 0 ... 29 29] = Grid position of sized-down image (same size as descriptors)
        #             [ 0 1 2 ... 38 39]
        fmap_pos1 = grid_positions(h1, w1, device)

        # pos1.shape = [2, 1200] = [2, 30x40]
        # pos1 = [ 3.5 3.5  ... 235.5 235.5] = Grid position (centers of tiles) of full-sized image
        #        [ 3.5 11.5 ... 307.5 315.5]
        pos1 = upscale_positions(fmap_pos1, scaling_steps=scaling_steps)


        #-----------------#
        # Correspondences #
        #-----------------#
        '''
        try:
            pos1, pos2, ids = warp(
                pos1,
                depth1, intrinsics1, pose1, bbox1,
                depth2, intrinsics2, pose2, bbox2
            )
        except EmptyTensorError:
            continue
        '''
        pos1, pos2, ids = warp(pos1, homography1, mask1, homography2, mask2)
       #-----------------------

        # Mask the (i) feature map positions (fmap_pos1) (ii) descriptors, (iii) scores
        # fmap_pos1.shape = [2, N]
        # descriptors1.shape = [512, N]
        # scores1.shape = [N]
        # N = # valid correspondences
        fmap_pos1 = fmap_pos1[:, ids]
        descriptors1 = descriptors1[:, ids]
        scores1 = scores1[ids]

        # Skip the pair if not enough GT correspondences are available
        if ids.size(0) < 128:
            continue

        # Construct fmap_pos2, descriptors2, so order of corresponding points matches fmap_pos1, descriptors1
        # fmap_pos2.shape = [2, N]
        # descriptors2.shape = [512, N]
        fmap_pos2 = torch.round(downscale_positions(pos2, scaling_steps=scaling_steps)).long()
        descriptors2 = F.normalize(dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]],dim=0)

        #------- Positive Distance -------#
        # positive_distance.shape = [N]
        positive_distance = 2 - 2 * (descriptors1.t().unsqueeze(1) @ descriptors2.t().unsqueeze(2)).squeeze()

        #------- Negative Distance: Image 2 ------#
        # all_fmap_pos2.shape = [2, HxW] = [2, 1200]
        all_fmap_pos2 = grid_positions(h2, w2, device)

        #--- ADDED: ONLY include valid descriptors in triplet loss
        # Get subset of 'all' positions that are valid
        pos2_all_valid = upscale_positions(all_fmap_pos2, scaling_steps=scaling_steps)
        pos2_all_valid, ids2_valid = apply_mask(pos2_all_valid, mask2)
        # Mask fmap and descriptor
        all_fmap_pos2 = torch.round(downscale_positions(pos2_all_valid, scaling_steps=scaling_steps)).long()
        all_descriptors2 = all_descriptors2[:, ids2_valid]
        #---

        # position_distance.shape = [N, HxW] = [N, 1200]
        # Distance between {descriptor2 with correspondence} and {all descriptor2}
        # First gets both x & y distance, then takes the maximum
        position_distance = torch.max( torch.abs(fmap_pos2.unsqueeze(2).float() - all_fmap_pos2.unsqueeze(1)), dim=0)[0]

        # If the *larger* position value is greater than the radius
        # is_out_of_safe_radius.shape = [N,HxW]
        is_out_of_safe_radius = position_distance > safe_radius
        
        # distance_matrix.shape = [N,HxW]
        distance_matrix = 2 - 2 * (descriptors1.t() @ all_descriptors2)

        negative_distance2 = torch.min( distance_matrix + (1 - is_out_of_safe_radius.float()) * 10., dim=1)[0]

        #------- Negative Distance: Image 1 ------#
        all_fmap_pos1 = grid_positions(h1, w1, device)

        #--- ADDED: ONLY include valid descriptors in triplet loss
        # Get subset of 'all' positions that are valid
        pos1_all_valid = upscale_positions(all_fmap_pos1, scaling_steps=scaling_steps)
        pos1_all_valid, ids1_valid = apply_mask(pos1_all_valid, mask1)
        # Mask fmap and descriptor
        all_fmap_pos1 = torch.round(downscale_positions(pos1_all_valid, scaling_steps=scaling_steps)).long()
        all_descriptors1 = all_descriptors1[:, ids1_valid]
        #--- 

        position_distance = torch.max( torch.abs(fmap_pos1.unsqueeze(2).float() - all_fmap_pos1.unsqueeze(1)), dim=0)[0]

        is_out_of_safe_radius = position_distance > safe_radius

        distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)

        negative_distance1 = torch.min(distance_matrix + (1 - is_out_of_safe_radius.float()) * 10., dim=1)[0]

        #------- Construct loss ------#
        # Part 1: diff
        diff = positive_distance - torch.min( negative_distance1, negative_distance2 )

        # Construct scores 2 (in order such that corresponding points line up with scores1)
        scores2 = scores2[fmap_pos2[0, :], fmap_pos2[1, :]]

        # Part 2: Loss for index
        # ORIGINAL: SOFT-SCORE WEIGHTED MARGIN LOSS (meant to enhance repeatability)
        # Added 1e-5 to denominator to avoid NaN (recommended by author in github issues)
        if weighted_loss:
            loss = loss + ( torch.sum(scores1 * scores2 * F.relu(margin + diff)) / (torch.sum(scores1 * scores2) + 1e-5))
        # NEW ATTEMPT: REGULAR MARGIN LOSS
        else:
            loss = loss + torch.sum(F.relu(margin + diff))

        has_grad = True
        n_valid_samples += 1

        if plot and batch['batch_idx'] % batch['log_interval'] == 0:
            pos1_aux = pos1.cpu().numpy()
            pos2_aux = pos2.cpu().numpy()
            k = pos1_aux.shape[1]
            col = np.random.rand(k, 3)
            n_sp = 4
            plt.figure()
            # Image 1
            plt.subplot(1, n_sp, 1)
            im1 = imshow_image(
                batch['image1'][idx_in_batch].cpu().numpy(),
                preprocessing=batch['preprocessing']
            )
            # Mask images before displaying
            im1 *= mask1[:,:,np.newaxis].cpu().numpy().astype(np.uint8)
            plt.imshow(im1)
            plt.scatter(
                pos1_aux[1, :], pos1_aux[0, :],
                s=0.25**2, c=col, marker=',', alpha=0.5
            )
            plt.axis('off')
            # Image 1 Scores
            plt.subplot(1, n_sp, 2)
            plt.imshow(
                output['scores1'][idx_in_batch].data.cpu().numpy(),
                cmap='Reds'
            )
            plt.axis('off')
            # Image 2
            plt.subplot(1, n_sp, 3)
            im2 = imshow_image(
                batch['image2'][idx_in_batch].cpu().numpy(),
                preprocessing=batch['preprocessing']
            )
            # Mask images before displaying
            im2 *= mask2[:,:,np.newaxis].cpu().numpy().astype(np.uint8)
            plt.imshow(im2)
            plt.scatter(
                pos2_aux[1, :], pos2_aux[0, :],
                s=0.25**2, c=col, marker=',', alpha=0.5
            )
            plt.axis('off')
            # Image 2 scores
            plt.subplot(1, n_sp, 4)
            plt.imshow(
                output['scores2'][idx_in_batch].data.cpu().numpy(),
                cmap='Reds'
            )
            plt.axis('off')
            fig_path = os.path.join(plot_path, '%s.%02d.%02d.%d.png' % ('train' if batch['train'] else 'valid', batch['epoch_idx'], batch['batch_idx'] // batch['log_interval'], idx_in_batch))
            savefig(fig_path, dpi=300)
            plt.close()

    if not has_grad:
        import ipdb; ipdb.set_trace()
        raise NoGradientError

    # Part 3: Average
    loss = loss / n_valid_samples

    return loss

# Checks that each pixel in 8x8 tile around 'pos' elements are valid
# Similar function to interpolated_depth
def apply_mask(pos, mask):
    # Set device
    device = pos.device

    # Get boundaries
    h, w = mask.shape

    # Search in an 8x8 grid about each position
    # If even one pixel is invalid, the whole position is invalid
    valid_points = [] # Boolean list to mask ids
    for point in pos.T: 
        # Start with true
        valid = True

        # Get point
        i = round(point[0].item())
        j = round(point[1].item())

        # If not in-bounds, discard
        if not ((i >= 4) and (i < (h-4)) and (j >= 4) and (j < (w-4))): valid *= False

        # Check if *all* elements in 8x8 tile 'True' in valid mask
        if not torch.all(mask[i-4:i+4,j-4:j+4]): valid *= False

        # Append to points
        valid_points.append(int(valid))

    # Take the nonzero elements as the ids
    ids = torch.nonzero( torch.tensor(valid_points))[:,0].to(device)

    # Apply to pos
    pos = pos[:,ids]

    return pos, ids

# Overwriting original warp function
def warp(pos1, homography1, mask1, homography2, mask2):

    device = pos1.device

    # Get valid positions in image 1
    pos1, ids = apply_mask(pos1, mask1)

    #----- Warp -----#
    # Get in homogeneous format
    # Change from [2, H x W] to [3, H x W] & convert from x,y to u,v
    pos1_aug = torch.vstack((pos1[1,:], pos1[0,:], torch.ones((pos1.shape[1]), device=device)))
    # Apply homographies to warp pos1 to image 2
    homography = torch.mm(homography2, torch.inverse(homography1))
    pos2_aug   = torch.mm(homography, pos1_aug)
    # Remove from homogeneous format (divide through by last element)
    # Change from [3, H x W] to [2, H x W], convert back to x,y from u,v
    pos2 = torch.vstack((pos2_aug[1],pos2_aug[0])) / pos2_aug[2]

    # Get valid positions in image 2
    pos2, new_ids = apply_mask(pos2, mask2)

    # Update the valid positions in image 1
    ids = ids[new_ids]
    pos1 = pos1[:, new_ids]

    return pos1, pos2, ids

#------------------------#
# OLD FUNCTIONS NOT USED # 
#------------------------#
def interpolate_depth(pos, depth):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    h, w = depth.size()

    i = pos[0, :]
    j = pos[1, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Valid depth
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    ids = ids[valid_depth]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]

def uv_to_pos(uv):
    return torch.cat([uv[1, :].view(1, -1), uv[0, :].view(1, -1)], dim=0)
