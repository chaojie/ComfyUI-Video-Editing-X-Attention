from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor, Attention
import torch.nn.functional as nnf
import numpy as np
import sys
import abc
import fastcore.all as fc
import math
from skimage.draw import disk
import torch.nn.functional as F
from functools import partial
from .utils.guidance_functions import *
from .ptp_utils import text2video
import numpy
from compel import Compel
import diffusers
import matplotlib.pyplot as plt
from diffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline, UNet3DConditionModel
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from einops import rearrange
from huggingface_hub import snapshot_download
import sys

from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import pathlib

from PIL import Image


def normalize(x): return (x - x.min()) / (x.max() - x.min())
def threshold_attention(attn, s=10):
    norm_attn = s * (normalize(attn) - 0.5)
    return normalize(norm_attn.sigmoid())

def get_shape(attn, s=20): 
    return threshold_attention(attn, s)

def get_size(attn): 
    return 1/attn.shape[-2] * threshold_attention(attn).sum((1,2)).mean()

def enlarge(x, scale_factor=1.0):
    x = x.view(1, -1, 1)
    assert scale_factor >= 1

    h = w = int(math.sqrt(x.shape[-2]))
    x = rearrange(x, 'n (h w) d -> n d h w', h=h)
    x = F.interpolate(x, scale_factor=scale_factor)
    new_h = new_w = x.shape[-1]
    x_l, x_r = (new_w//2) - w//2, (new_w//2) + w//2
    x_t, x_b = (new_h//2) - h//2, (new_h//2) + h//2
    x = x[:,:,x_t:x_b,x_l:x_r]
    return rearrange(x, 'n d h w -> n (h w) d', h=h) * scale_factor

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def run_and_display(pipe, prompt, obj_to_edit, objects, guidance_func,  latent=None, run_baseline=False, width=256, height=256, num_frames=16, num_inference_steps=50, guidance_scale=15, seed=1234):
    videos, orig_video, x_t = text2video(pipe, prompt, obj_to_edit, objects, guidance_func,width=width,height=height,num_frames=num_frames,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,seed=seed) #num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, low_resource=LOW_RESOURCE)
    
    return videos, orig_video, x_t

class VEXALoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("VEXAPipe",)
    FUNCTION = "run"
    CATEGORY = "VEXA"

    def run(self):
        with torch.inference_mode(False):
            numpy.set_printoptions(threshold=sys.maxsize)
            LOW_RESOURCE = False 
            NUM_DIFFUSION_STEPS = 50
            GUIDANCE_SCALE = 7.5
            MAX_NUM_WORDS = 77
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
            #ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN).to(device)
            #tokenizer = ldm_stable.tokenizer
            #pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
            #pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
            #pipe.enable_model_cpu_offload()
            pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()
            pipe.to(device)

        return (pipe,)

class VEXAGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "direction": (["up","down","right","left"], {"default": "right"}),
                "factor":("FLOAT",{"default":.1}),
                "position_weight":("INT",{"default":4}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("VEXAGuidance",)
    FUNCTION = "run"
    CATEGORY = "VEXA"

    def run(self,direction,factor,position_weight):
        with torch.inference_mode(False):
            move = partial(roll_shape, direction=direction, factor=factor)
            guidance = partial(edit_by_E, shape_weight=0, appearance_weight = 0, position_weight=4, tau=move)
            return (guidance,)

class StringList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": ""}),
                "string2": ("STRING", {"default": ""}),
                "string3": ("STRING", {"default": ""}),
                "string4": ("STRING", {"default": ""}),
                "string5": ("STRING", {"default": ""}),
                "string6": ("STRING", {"default": ""}),
            },
            "optional": {
                "prev_strings": ("StringList",)
            }
        }

    RETURN_TYPES = ("StringList",)
    FUNCTION = "run"
    CATEGORY = "VEXA"

    def run(self, string1: str, string2: str, string3: str, string4: str, string5: str, string6:str, prev_strings: list[str]=None):
        with torch.inference_mode(False):
            strings=[x for x in [string1, string2, string3, string4, string5, string6] if x!='']
            if prev_strings is not None:
                return (prev_strings+strings,)
            return (strings,)


class VEXARun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("VEXAPipe",),
                "prompt": ("STRING", {"default": "darth vader surfs in the ocean"}),
                "obj_to_edit": ("STRING", {"default": "darth"}),
                "objects": ("StringList", {"default": ['darth', 'ocean']}),
                "guidance":("VEXAGuidance",),
                "width":("INT",{"default":256}),
                "height":("INT",{"default":256}),
                "num_frames":("INT",{"default":16}),
                "num_inference_steps":("INT",{"default":50}),
                "guidance_scale":("FLOAT",{"default":15.0}),
                "seed":("INT",{"default":1234}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("video","orig_video",)
    FUNCTION = "run"
    CATEGORY = "VEXA"

    def run(self,pipe,prompt,obj_to_edit,objects,guidance,width,height,num_frames,num_inference_steps,guidance_scale,seed):
        with torch.inference_mode(False):
        #    with torch.set_grad_enabled(False):
                prompts = [prompt]
                tokens = [2]
                videos, orig_video, x_t  = run_and_display(pipe, prompts, obj_to_edit, objects, guidance_func=guidance,  latent=None, run_baseline=False,width=width,height=height,num_frames=num_frames,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,seed=seed)

                xvideo=None
                ovideo=None
                for video in [videos[0]]:
                    xvideo = rearrange(video.cpu(), "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)

                for video in [orig_video[0]]:
                    ovideo = rearrange(video.cpu(), "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)

                return (xvideo,ovideo,)

    
NODE_CLASS_MAPPINGS = {
    "VEXALoader":VEXALoader,
    "VEXAGuidance":VEXAGuidance,
    "StringList":StringList,
    "VEXARun":VEXARun,
}
