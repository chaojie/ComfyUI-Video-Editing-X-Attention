#This file is the modified version of the same file name from: https://github.com/Sainzerjj/Free-Guidance-Diffusion/blob/master/free_guidance.py

import torch
from torch import tensor
from einops import rearrange
import numpy as np
import fastcore.all as fc
import math
import torch.nn.functional as F
from sklearn.decomposition import PCA
import torchvision.transforms as T
from PIL import Image
import os
from scipy.ndimage import label, generate_binary_structure
from copy import deepcopy
import copy
from functools import partial
# the calculation of character or attribute
import matplotlib.pyplot as plt



def keep_top_percent(tensor, percent=1):
    # Flatten the tensor
    flattened_tensor = tensor.flatten()
    
    # Sort the flattened tensor in descending order
    sorted_tensor, _ = torch.sort(flattened_tensor, descending=True)
    
    # Determine the threshold value for the top percentile
    percentile_index = int(len(sorted_tensor) * (percent / 100))
    threshold_value = sorted_tensor[percentile_index]
    
    # Reshape the threshold value to the shape of the original tensor
    threshold_tensor = threshold_value.expand_as(tensor)
    
    # Create a mask to zero out values below the threshold
    mask = tensor >= threshold_tensor
    
    # Apply the mask
    result_tensor = tensor * mask
    
    return result_tensor


#generate circular target cross-attention
def draw_circular_target_attention(tensor, height, width):
    # Calculate the centroid and radius of the circle
    centroid = ( 4.5 * height) // 9
    center_x =  ( 4.5 * width) // 9
    radius = height // 8

    # Create a grid of coordinates
    x_grid, y_grid = torch.meshgrid(torch.arange(height), torch.arange(width))
    
    # Calculate distances from each pixel to the centroid
    distances = torch.sqrt((x_grid - centroid)**2 + (y_grid - center_x)**2)
    
    # Create a mask for pixels within the circle
    circle_mask = distances <= radius
    
    # Set values inside the circle to 1 for all channels
    for i in range(tensor.size(0)):
        tensor[i, ~circle_mask.flatten()] = 0
        tensor[i, circle_mask.flatten()] = 1
    return tensor

#generate rectangular target cross-attention
def draw_rectangular_target_attention(tensor, height, width):
    start_x = int(2* width // 7)
    start_y = int(1 * height // 7)
    rect_height = 3 * int(height // 5)
    rect_width = 2 * int(width // 6)
    # Extract the number of channels
    num_channels = tensor.size(0)
    
    # Create a mask tensor of the same shape as the input tensor
    mask = torch.zeros_like(tensor.view(num_channels, height, width))
    
    # Determine the end coordinates of the rectangle
    end_x = min(start_x + rect_height, height)
    end_y = min(start_y + rect_width, width)
    
    # Iterate over each channel
    for c in range(num_channels):
        # Fill the rectangle area in the mask with 1s for each channel
        mask[c, start_x:end_x, start_y:end_y] = 1

    # Apply the mask to the input tensor
    tensor = tensor.view(num_channels, height, width) * (1 - mask)  # Set outside values to 0
    
    tensor += mask  # Set inner rectangle values to 1
    
    return tensor.view(num_channels, -1)


    return tensor


def gaussian_kernel(kernel_size, sigma):
    """
    Create a 2D Gaussian kernel
    """
    kernel = torch.tensor([[np.exp(-(x**2 + y**2) / (2. * sigma**2)) for x in range(-kernel_size//2 + 1, kernel_size//2 + 1)] for y in range(-kernel_size//2 + 1, kernel_size//2 + 1)])
    kernel = kernel / torch.sum(kernel)
    return kernel

def gaussian_smooth(input_tensor, kernel_size=3, sigma=1):
    """
    Apply Gaussian smoothing to a 2D tensor
    """
    # Create Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    # Add dimensions to the kernel for compatibility with conv2d function
    kernel = kernel.unsqueeze(0).unsqueeze(0).cuda()
    # Convert input tensor to float
    input_tensor = input_tensor.to(torch.float64).cuda()
    # Apply convolution
    smoothed = F.conv2d(input_tensor.unsqueeze(0), kernel, padding=kernel_size//2)
    return smoothed.squeeze(0)


def normalize(x): 
    return (x - x.min()) / (x.max() - x.min())

def filter_tensor_by_value(tensor):
    value = torch.max(tensor)
    filtered_tensor = torch.zeros_like(tensor)
    filtered_tensor[tensor == value] = value
    return filtered_tensor

def threshold_attention(attn, s=10):
    norm_attn = s * (normalize(attn) - 0.5)
    return normalize(norm_attn.sigmoid())

def get_shape(attn, s=20): 
    return threshold_attention(attn, s)

def get_size(attn): 
    return 1/attn.shape[-1] * threshold_attention(attn).sum((0,1)).mean()

def get_c(cross_attention_tensor):
 
    cross_attention_tensor = cross_attention_tensor
    x_coords, y_coords = torch.meshgrid(torch.arange(cross_attention_tensor.shape[0]),torch.arange(cross_attention_tensor.shape[1]))

    # Compute the weighted sum of x and y coordinates
    weighted_x = (x_coords.cuda() * cross_attention_tensor.cuda()).sum()
    weighted_y = (y_coords.cuda() * cross_attention_tensor.cuda()).sum()

    # Compute the total weight
    total_weight = cross_attention_tensor.sum().cuda()

    # Compute the centroid coordinates
    centroid_x = weighted_x / total_weight
    centroid_y = weighted_y / total_weight
    #print(torch.stack([centroid_x, centroid_y]))
    return torch.stack([centroid_x, centroid_y])
    
    #return centroid_x.cpu().item(), .cpu().item()

def get_centroid(attn):
    centeroid_per_frame_window = []
    if not len(attn.shape) == 3: attn = attn[:,:,None]
    h = w = int(tensor(attn.shape[-2]).sqrt().item())
    hs = torch.arange(h).view(-1, 1, 1).to(attn.device)
    ws = torch.arange(w).view(1, -1, 1).to(attn.device)
    attn = rearrange(attn.mean(0), '(h w) d -> h w d', h=h)
    weighted_w = torch.sum(ws * attn, dim=[0,1])
    weighted_h = torch.sum(hs * attn, dim=[0,1])
    return torch.stack([weighted_w, weighted_h]) / attn.sum((0,1))


def get_appearance(attn, feats):
    if not len(attn.shape) == 3: attn = attn[:,:,None]
    h = w = int(tensor(attn.shape[-2]).sqrt().item())
    shape = get_shape(attn).detach().mean(0).view(h,w,attn.shape[-1])
    feats = feats.mean((0,1))[:,:,None]
    return (shape * feats).sum() / shape.sum()

def get_attns(attn_storage):
    if attn_storage is not None:
        origs = attn_storage.maps('ori')
        #if "w" in origs.keys():
        #del(origs["w"])

        edits = attn_storage.maps('edit')
        #if "w" in edits.keys():
        #del(edits["w"])
    return origs, edits


def E(orig_attns, edit_attns, indices, tau, frame):
    shapes = []
    num_frames = 16
    for f in range(num_frames):
        deltas = []
        delta = torch.tensor(0).to(torch.float16).cuda()
        out = []
        i = 0
        
        for location in ["down","mid", "up"]:
            for o in indices:
                for edit_attn_map_integrated, ori_attn_map_integrated in zip(edit_attns[location], orig_attns[location]):
                    edit_attn_map = edit_attn_map_integrated.chunk(2)[1]
                    ori_attn_map = ori_attn_map_integrated.chunk(2)[1]
                    window_size = int(ori_attn_map.shape[0] // num_frames)
                    orig, edit = ori_attn_map[f * window_size:f * window_size + window_size,:,o], edit_attn_map[f * window_size:f * window_size + window_size,:,o]
                    h = w = int(tensor(orig.shape[1]).sqrt().item())
                    orig_copy = copy.deepcopy(orig)
                    ori = draw_circular_target_attention(orig, h, w)

                    #for 256 x 256 images, use 8 x 8 and 16 x 16 cross-attentions
                    if h in [16, 8]:
                        i += 1
                        if len(ori.shape) < 3: ori, edit = ori[...,None], edit[...,None]
                        #control how the target attention moves at each frame - through factor strength
                        rolled_attention = roll_shape(get_shape(ori), 'right', factor= 0.01 * f)
                        delta =(get_shape(rolled_attention) - get_shape(edit)).pow(2).mean()  
                    else:
                        delta = torch.tensor(0).to(torch.float16).cuda()
                    deltas.append(delta)
        shapes.append(torch.stack(deltas).mean())
    return torch.stack(shapes).sum()



#roll the target cross-attention -- used to change per-frame position
def roll_shape(x, direction='up', factor=0.5):
    h = w = int(math.sqrt(x.shape[-2]))
    mag = (0,0)
    if direction == 'up': mag = (int(-h*factor),0)
    elif direction == 'down': mag = (int(h*factor),0)
    elif direction == 'right': mag = (0,int(w*factor))
    elif direction == 'left': mag = (0,int(-w*factor))
    shape = (x.shape[0], h, h, x.shape[-1])
    x = x.view(shape)
    move = x.roll(mag, dims=(1,2))
    return move.view(x.shape[0], h*h, x.shape[-1])

def roll_shape_mag(x, mag, factor=0.5):
    h = w = int(math.sqrt(x.shape[-2]))
    shape = (x.shape[0], h, h, x.shape[-1])
    x = x.view(shape)
    move = x.roll(mag, dims=(1,2))
    return move.view(x.shape[0], h*h, x.shape[-1])


def edit_by_E(attn_storage, indices, tau=fc.noop, shape_weight=1, appearance_weight=1, position_weight=8, ori_feats=None, edit_feats=None, frame=None, **kwargs):
    origs, edits = get_attns(attn_storage)
    if len(indices) > 1: 
        obj_idx, other_idx = indices
        indices = torch.cat([obj_idx, other_idx])

    move_term = position_weight * E(origs, edits, obj_idx, tau=tau, frame=frame) 
    return move_term  

