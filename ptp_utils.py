#Created by Saman Motamed
#We build on the foolwoing works:
# https://github.com/google/prompt-to-prompt
# https://github.com/Sainzerjj/Free-Guidance-Diffusion/tree/master

import numpy as np
import torch
import diffusers
from diffusers.models.attention_processor import AttnProcessor, Attention
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import argparse
import os
import platform
import torch, gc
from diffusers.loaders import TextualInversionLoaderMixin
import re
import warnings
from pathlib import Path
from typing import List, Optional
from uuid import uuid4
import fastcore.all as fc
from functools import partial
import numpy as np
import torch
from compel import Compel
from diffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline, UNet3DConditionModel
from einops import rearrange
from torch import Tensor
from torch.nn.functional import interpolate
from tqdm import trange
from copy import deepcopy
import copy
from .train import export_to_video, handle_memory_attention, load_primary_models
from .utils.lama import inpaint_watermark
from .utils.lora import inject_inferable_lora
from diffusers.models.attention_processor import AttnProcessor, Attention
import torch.nn.functional as F
import math
from torch import tensor
from diffusers import LMSDiscreteScheduler, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from .utils.lora_handler import LoraHandler
import imageio
import comfy.utils



random_seed = 2322
torch.manual_seed(random_seed)
def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    #latents = controller.step_callback(latents)
    return latents


def get_features(hook, layer, inp, out):
    if not hasattr(hook, 'feats'): hook.feats = out
    hook.feats = out

class Hook():
    def __init__(self, model, func): self.hook = model.register_forward_hook(partial(func, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()




class AttentionStoreSam:
    @staticmethod
    def get_empty_store():
        return {'ori' : {"down": [], "mid": [], "up": [], "w" :[]}, 'edit' :  {"down": [], "mid": [], "up": [], "w":[]}}
    def __init__(self, attn_res=[1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]): 
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res

    def __call__(self, attention_map, is_cross, place_in_unet: str, pred_type='ori'): 
        # if not name in self.step_store: 
        #     self.step_store[name] = {}
        # self.step_store[name][pred_type] = attention_map
        if self.cur_att_layer >= 0 and is_cross:
            if attention_map.shape[1] in self.attn_res:
                self.step_store[pred_type][place_in_unet].append(attention_map)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps(pred_type)
    
    def aggregate_attention(self, from_where: List[str]) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[location]:
                cross_maps = item.reshape(-1, self.attn_res[0], self.attn_res[1], item.shape[-1])
                out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
    
    def maps(self, block_type: str):
        return self.attention_store[block_type]

    def between_steps(self, pred_type='ori'):
        self.attention_store[pred_type] = self.step_store[pred_type]
        self.step_store = self.get_empty_store()

        
class CustomAttnProcessor(AttnProcessor):
    def __init__(self, attnstore, place_in_unet=None): 
        super().__init__()
        fc.store_attr()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.store = False

    def set_storage(self, store, pred_type): 
        self.store = store
        self.pred_type = pred_type

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
     
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        if self.store: 
            self.attnstore(attention_probs, is_cross, self.place_in_unet, pred_type=self.pred_type) ## stores the attention maps in attn_storage
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states



def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


############################################ TEXT TO VIDEO ##############################################

def prepare_input_latents(
    pipe: TextToVideoSDPipeline,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    init_video: Optional[str],
    vae_batch_size: int,
):
    if init_video is None:
        # initialize with random gaussian noise
        scale = pipe.vae_scale_factor
        shape = (batch_size, pipe.unet.config.in_channels, num_frames, height // scale, width // scale)
        latents = torch.randn(shape, dtype=torch.half)

    else:
        # encode init_video to latents
        latents = encode(pipe, init_video, vae_batch_size)
        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)

    return latents



def encode(pipe: TextToVideoSDPipeline, pixels: Tensor, batch_size: int = 8):
    nf = pixels.shape[2]
    pixels = rearrange(pixels, "b c f h w -> (b f) c h w")

    latents = []
    for idx in trange(
        0, pixels.shape[0], batch_size, desc="Encoding to latents...", unit_scale=batch_size, unit="frame"
    ):
        pixels_batch = pixels[idx : idx + batch_size].to("cuda", dtype=torch.half)
        latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample().cuda()
        latents_batch = latents_batch.mul(pipe.vae.config.scaling_factor)
        latents.append(latents_batch)
    latents = torch.cat(latents)

    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=nf)

    return latents


def decode(pipe: TextToVideoSDPipeline, latents: Tensor, batch_size: int = 8):
    nf = latents.shape[2]
    latents = rearrange(latents, "b c f h w -> (b f) c h w")

    pixels = []
    for idx in trange(
        0, latents.shape[0], batch_size, desc="Decoding to pixels...", unit_scale=batch_size, unit="frame"
    ):
        latents_batch = latents[idx : idx + batch_size].to("cuda", dtype=torch.half)
        latents_batch = latents_batch.div(pipe.vae.config.scaling_factor)
        pixels_batch = pipe.vae.decode(latents_batch).sample.cuda()
        pixels.append(pixels_batch)
    pixels = torch.cat(pixels)

    pixels = rearrange(pixels, "(b f) c h w -> b c f h w", f=nf)

    return pixels.float()




def prepare_attention(unet, pred_type, set_store=True):
        for name, module in unet.attn_processors.items():
            module.set_storage(set_store, pred_type)

def sample(unet, latents, scheduler, t, feature_layer, guidance_scale, cond_prompt_embeds, prompt_embeds, hook=None, pred_type='edit', set_store=True, do_classifier_free_guidance=True):   

        #feature_layer = unet.up_blocks[-1].resnets[-2]

        if True:
            if pred_type == 'ori':
                latent_model_input = scheduler.scale_model_input(
                latents, t
                )
                prepare_attention(unet, pred_type=pred_type, set_store=set_store)
                
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=cond_prompt_embeds
                ).sample

                feats = hook.feats if feature_layer is not None else None
                if pred_type == 'edit':
                    unet.zero_grad()
                # perform guidance
                gc.collect()
                torch.cuda.empty_cache()
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = scheduler.scale_model_input(
                        latent_model_input, t
                    )

            # predict the noise residual
           
            with torch.no_grad():
                noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds
                ).sample
           

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            feats = None
            return noise_pred, feats
 


def video_diffusion_step(pipe, model_outputs, prompt_embeds, controller, latents, window_size, rotate, t, i, guidance_scale, low_resource=False):
    
    edit_scheduler = copy.deepcopy(pipe.scheduler)

    order = pipe.scheduler.config.solver_order if "solver_order" in pipe.scheduler.config else pipe.scheduler.order
    do_classifier_free_guidance = guidance_scale > 1.0


    batch_size, _, num_frames, _, _ = latents.shape
    device = pipe.device

    if rotate:
        shifts = np.random.permutation(primes_up_to(window_size))
        total_shift = 0

    #latents = latents.clone().detach()
    #latents.requires_grad_(True).to(torch.float32)

    new_latents = torch.zeros_like(latents)
    new_outputs = torch.zeros_like(latents)

    for idx in range(0, num_frames, window_size):  # diffuse each chunk individually
        # update scheduler's previous outputs from our own cache
        pipe.scheduler.model_outputs = [model_outputs[(i - 1 - o) % order] for o in reversed(range(order))]
        pipe.scheduler.model_outputs = [
            None if mo is None else mo[:, :, idx : idx + window_size, :, :].to("cuda")
            for mo in pipe.scheduler.model_outputs
        ]
        pipe.scheduler.lower_order_nums = min(i, order)

        #latents_window = latents[:, :, idx : idx + window_size, :, :].to("cuda")

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        #noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
        
        
        #noise_pred, loss = some_loss(pipe, controller, latent_model_input, t, prompt_embeds)
        

        ############################### guidance comes in play #####################################33



        with torch.no_grad():

            


            #latents_window = latents#[:, :, idx : idx + window_size, :, :].to("cuda")

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
   


            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
            # reshape latents for scheduler
            pipe.scheduler.model_outputs = [
                None if mo is None else rearrange(mo, "b c f h w -> (b f) c h w")
                for mo in pipe.scheduler.model_outputs
            ]

            
            latents = rearrange(latents, "b c f h w -> (b f) c h w").cuda()
            noise_pred = rearrange(noise_pred, "b c f h w -> (b f) c h w")
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = pipe.scheduler.step(noise_pred.cuda(), t, latents).prev_sample
            
            # reshape latents back for UNet
            latents = rearrange(latents, "(b f) c h w -> b c f h w", b=batch_size)
            
            # write diffused latents to output
            new_latents[:, :, idx : idx + window_size, :, :] = latents.cuda()
    

            # store scheduler's internal output representation in our cache
            '''new_outputs[:, :, idx : idx + window_size, :, :] = rearrange(
                pipe.scheduler.model_outputs[-1], "(b f) c h w -> b c f h w", b=batch_size
            )
            


        
        model_outputs[i % order] = new_outputs'''

        latents = new_latents

        return latents.to(device)


def primes_up_to(n):
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n**0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3 :: 2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]



def choose_object_indexes(model, prompt, objects:list=None, object_to_edit=None):
        """Extracts token indexes only for user-defined objects."""

        prompt_inputs = model.tokenizer(prompt, padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids

        if object_to_edit is not None: 
            
            obj_inputs = model.tokenizer(object_to_edit, add_special_tokens=False).input_ids
            

            obj_idx = torch.cat([torch.where(prompt_inputs == o)[1] for o in obj_inputs])
            
            if object_to_edit in objects: objects.remove(object_to_edit)
        other_idx = []

        for o in objects:
            
            inps = model.tokenizer(o, add_special_tokens=False).input_ids
            
            other_idx.append(torch.cat([torch.where(prompt_inputs == o)[1] for o in inps]))
            
        if object_to_edit is None: 
            return torch.cat(other_idx)
        else: 
            return obj_idx, torch.cat(other_idx)


def all_word_indexes(model, prompt, object_to_edit=None, **kwargs):
        
        """Extracts token indexes by treating all words in the prompt as separate objects."""
        prompt_inputs = model.tokenizer(prompt, padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
        if object_to_edit is not None: 
            obj_inputs = model.tokenizer(object_to_edit, add_special_tokens=False).input_ids
            
            obj_idx = torch.cat([torch.where(prompt_inputs == o)[1] for o in obj_inputs])
            a = set([i for i, o in enumerate(prompt_inputs[0]) if o not in obj_inputs])
            b = set(torch.where(prompt_inputs < 49405)[1].numpy())
            other_idx = tensor(list(a&b))
            return obj_idx, other_idx
        else: 
            return torch.where(prompt_inputs < 49405)[1]
        
def encode_prompt(
        model,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(model, TextualInversionLoaderMixin):
                prompt = model.maybe_convert_prompt(prompt, model.tokenizer)

            text_inputs = model.tokenizer(
                prompt,
                padding="max_length",
                max_length=model.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = model.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = model.tokenizer.batch_decode(
                    untruncated_ids[:, model.tokenizer.model_max_length - 1 : -1]
                )


            if (
                hasattr(model.text_encoder.config, "use_attention_mask")
                and model.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = model.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=model.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(model, TextualInversionLoaderMixin):
                uncond_tokens = model.maybe_convert_prompt(uncond_tokens, model.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = model.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(model.text_encoder.config, "use_attention_mask")
                and model.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = model.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=model.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            final_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return final_prompt_embeds, prompt_embeds




#guidance schedule for updating z at specified steps
def do_self_guidance(model, time, T, scheduler):
        if type(scheduler).__name__ == "DDPMScheduler":
            if time <= int((4*T)/16): return True
            elif time >= int(T - T/32): return False
            elif time % 2 == 0: return True
            else: return False
        if type(scheduler).__name__ == "DDIMScheduler":
            if time <= int((3*T)/16): return True
            elif time >= int(T - T/32): return False
            elif time % 2 == 0: return True
            else: return False
        elif type(scheduler).__name__ == "LMSDiscreteScheduler":
            if time <= int(T/5): return True
            elif time >= T - 5: return False
            elif time % 2 == 0: return True
            else: return False
        elif type(scheduler).__name__ == "DPMSolverMultistepScheduler":
            if time <= int(2*T/5): return True
            elif time >= T - 5: return False
            elif time % 2 == 0: return True
            else: return False


def text2video(
    model: TextToVideoSDPipeline,
    prompt: Optional[List[str]],
    obj_to_edit,
    objects,
    guidance_func,
    width=256,
    height=256,
    num_frames=16,
    num_inference_steps=50,
    guidance_scale=15,
    seed=1234,
):
    with torch.no_grad():
        feature_layer = None
        negative_prompt = None
        negative_prompt_embeds =None
        #num_inference_steps = 50
        #guidance_scale = 15
        rotate = False
        low_resource = False
        xformers = False
        sdp = False
        lora_path= ""
        lora_rank = 64
        loop = False
        #seed = None
        #width = 256
        #height= 256
        #num_frames= 16
        window_size = 48
        vae_batch_size = 2
        init_video=None
        init_weight = 0
        device = model.device
        num_images_per_prompt = 1
        feature_layer = model.unet.up_blocks[-1].resnets[-2]

        max_guidance_iter_per_step: int = 1
        order = model.scheduler.config.solver_order if "solver_order" in model.scheduler.config else model.scheduler.order
        do_classifier_free_guidance = guidance_scale > 1.0
        ori_prompt = None
        prompt_embeds = None
        window_size = min(num_frames, window_size)

        latents = prepare_input_latents(
                pipe=model,
                batch_size=len(prompt),
                num_frames=num_frames,
                height=height,
                width=width,
                init_video=init_video,
                vae_batch_size=vae_batch_size,
            )
        batch_size, _, num_frames, _, _ = latents.shape

        #compel = Compel(tokenizer=model.tokenizer, text_encoder=model.text_encoder)
        #prompt_embeds, negative_prompt_embeds = compel(prompt), compel(negative_prompt) if negative_prompt else None

        prompt_embeds, cond_prompt_embeds = encode_prompt(
            model,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )


        if ori_prompt is None :
            ori_prompt = prompt
        ori_prompt_embeds, ori_cond_prompt_embeds = encode_prompt(
            model,
            ori_prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # set the scheduler to start at the correct timestep
        model.scheduler.set_timesteps(num_inference_steps, device="cuda")
    
        start_step = round(init_weight * len(model.scheduler.timesteps))
        timesteps = model.scheduler.timesteps[start_step:]
        if init_weight == 0:
            torch.manual_seed(seed)
            latents = torch.randn_like(latents, memory_format=torch.contiguous_format)
        else:
            latents = model.scheduler.add_noise(
                original_samples=latents, noise=torch.randn_like(latents), timesteps=timesteps[0]
            )
    
        # manually track previous outputs for the scheduler as we continually change the section of video being diffused

        ####################### taken from free muidance #####################
        unet = model.unet

        spatial_lora_path: str = ""
        temporal_lora_path: str = ""
        #lora_rank: int = 64,
        spatial_lora_scale: float = 1.0
        lora_scale: float = 1.0

        lora_manager_temporal = LoraHandler(
        version="cloneofsimo",
        use_unet_lora=True,
        use_text_lora=False,
        save_for_webui=False,
        only_for_webui=False,
        unet_replace_modules=["TransformerTemporalModel"],
        text_encoder_replace_modules=None,
        lora_bias=None
        )


        #enabling LoRA MOtionDirector guidance...In the works
        #unet_lora_params, unet_negation = lora_manager_temporal.add_lora_to_model(
        #True, unet, lora_manager_temporal.unet_replace_modules, 0, lora_path, r=lora_rank, scale=lora_scale)
     



        attention_store = AttentionStoreSam()
        with torch.enable_grad():
            register_attention_control(unet, attention_store)
        g_name = guidance_func.func.__name__ if isinstance(guidance_func, partial) else guidance_func.__name__
        
        if g_name not in ['edit_appearance'] and feature_layer is None:
            
            feature_layer = unet.up_blocks[-1].resnets[-2]
        feature_layer = unet.up_blocks[-1].resnets[-1]
        if feature_layer is not None: 
            hook = Hook(feature_layer, get_features)
        else: 
            hook = None


        if all_word_indexes.__name__ == 'choose_object_indexes' and objects is None:
            raise ValueError('Provide a list of object strings from the prompt.')
        if g_name not in ['edit_layout', 'edit_appearance', 'edit_layout_by_feature'] and obj_to_edit is None:
            raise ValueError('Provide an object string for editing.')
        if obj_to_edit is None:
            
            indices = all_word_indexes(model, prompt, objects=objects, object_to_edit=obj_to_edit)
        else:
            
            indices = choose_object_indexes(model, prompt, objects=objects, object_to_edit=obj_to_edit)

        ori_latents = latents.clone().detach().cuda()
        edit_latent = latents.clone().detach().cuda()
  
        model_outputs = [None] * order
        edit_scheduler = copy.deepcopy(model.scheduler)
        pbar = comfy.utils.ProgressBar(len(timesteps))
        for i, t in enumerate(timesteps):        
            prepare_attention(unet, pred_type='ori', set_store=True)        
            ori_noise_pred = unet(ori_latents,t,encoder_hidden_states=ori_cond_prompt_embeds).sample
            ori_feats = hook.feats if feature_layer is not None else None
            
            with torch.enable_grad():
                if do_self_guidance(model, i, len(model.scheduler.timesteps), model.scheduler):
                    for guidance_iter in range(1):
                        latents = latents.requires_grad_(True).to("cuda")
                        latent_model_input = latents
                        latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)                
                        prepare_attention(unet, pred_type='edit', set_store=True)
                        edit_noise_pred = unet(latent_model_input,t,encoder_hidden_states=cond_prompt_embeds).sample    
                        edit_feats = hook.feats if feature_layer is not None else None
                        loss = guidance_func(attention_store, indices, ori_feats=ori_feats, edit_feats=edit_feats)
                        grad_cond = torch.autograd.grad(
                            loss.requires_grad_(True),
                            [latents],
                            retain_graph=True,
                        )[0]
                        print("loss :", loss)
                        if isinstance(model.scheduler, LMSDiscreteScheduler):
                            sig_t = model.scheduler.sigmas[i]
                        else:
                            sig_t = 1 - model.scheduler.alphas_cumprod[t]
                        backward_guidance_scale = 15
                        latents = latents - backward_guidance_scale * sig_t * grad_cond
                        torch.cuda.empty_cache()
                with torch.no_grad():
                    ori_latent_model_input = (torch.cat([ori_latents] * 2) if do_classifier_free_guidance else ori_latents)
                    ori_latent_model_input = model.scheduler.scale_model_input(ori_latent_model_input , t)
                    ori_edit_noise_pred= unet(ori_latent_model_input,t,encoder_hidden_states=ori_prompt_embeds).sample
                    if do_classifier_free_guidance:
                        ori_noise_pred_uncond, ori_noise_pred_text = ori_edit_noise_pred.chunk(2)
                        ori_edit_noise_pred = ori_noise_pred_uncond + guidance_scale * (
                            ori_noise_pred_text - ori_noise_pred_uncond
                        )
                    latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
                    latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
                    prepare_attention(unet, pred_type='edit', set_store=True)
                    edit_noise_pred= unet(latent_model_input,t,encoder_hidden_states=prompt_embeds).sample
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = edit_noise_pred.chunk(2)
                        edit_noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    latents =  edit_scheduler.step(edit_noise_pred, t, latents).prev_sample
                    ori_latents = model.scheduler.step(ori_edit_noise_pred, t, ori_latents).prev_sample
                    gc.collect()
                    torch.cuda.empty_cache()
            pbar.update(1)
            gc.collect()
            torch.cuda.empty_cache()

        
        orig_vid = decode(model, ori_latents, vae_batch_size)
        videos = decode(model, latents, vae_batch_size)
           
    
        return videos, orig_vid, latents
       

def register_attention_control(unet, attention_store):
        attn_procs = {}
        cross_att_count = 0
        for name in unet.attn_processors.keys():

            if "mid" in name:
                    place_in_unet = "mid"
                    cross_att_count += 1
                    attn_procs[name] = CustomAttnProcessor(attnstore=attention_store, place_in_unet=place_in_unet)
            elif "up" in name:
                    place_in_unet = "up"
                    cross_att_count += 1
                    attn_procs[name] = CustomAttnProcessor(attnstore=attention_store, place_in_unet=place_in_unet)
            elif "down" in name:
                    place_in_unet = "down"
                    cross_att_count += 1
                    attn_procs[name] = CustomAttnProcessor(attnstore=attention_store, place_in_unet=place_in_unet)
            else:
                place_in_unet = "w"
                cross_att_count += 1

                attn_procs[name] = CustomAttnProcessor(attnstore=attention_store, place_in_unet="w") 

        unet.set_attn_processor(attn_procs)
        attention_store.num_att_layers = cross_att_count
