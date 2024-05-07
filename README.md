# Investigating the Effectiveness of Cross Attention to Unlock Zero-Shot Editing of Text-to-Video Diffusion Models
## CVPR 2024 Generative Models for Computer Vision Workshop
<figures>
    <div>
      <img src="resources/video-edit-x-att (2).svg" width="1240" height="410" >  
    </div>
    <figcaption>From left to right; the original video generated with the caption "A burger floats on the water". The middle shows the target cross-attention maps to edit the burger so that it moves from the top-left to bottom-left of the scene. After updating the latent, the last video is generated capturing the intended edit.</figcaption>
</figures>

## Abstract
 With recent advances in image and video diffusion models for content creation, a plethora of techniques have been proposed for customizing their generated content. 
In particular, manipulating the cross-attention layers of Text-to-Image (T2I) diffusion models has shown great promise in controlling the shape and location of objects in the scene. Transferring image-editing techniques to the video domain, however, is extremely challenging as object motion and temporal consistency are difficult to capture accurately. In this work, we take a first look at the role of cross-attention in Text-to-Video (T2V) diffusion models for zero-shot video editing. While one-shot models have shown potential in controlling motion and camera movement, we demonstrate zero-shot control over object shape, position and movement in T2V models. We show that despite the limitations of current T2V models, cross-attention guidance can be a promising approach for editing videos.

## Setup
To set-up your environment and download one of the T2V models that we work with, please do these steps:
```
conda create -n video-edit python=3.10
conda activate video-edit
pip install -r requirements.txt

git clone https://github.com/sam-motamed/Video-Editing-X-Attention.git
cd Video-Editing-X-Attention
git lfs install
git clone https://huggingface.co/damo-vilab/text-to-video-ms-1.7b ./models/model_scope_diffusers/
```
## Editing Videos
You can edit an oject in a video via the jupyter notebook.
Simply provide a prompt and set ```obj_to_edit``` to a word in the prompt you want to edit.
To control the size and motion of the object, refer to ```function E``` in ```guidance_function.py```.
Experiment with different backward guidance scales and prompts to get the desired edit.
Note that current T2V models produce noisy cross-attentions so if your edit doesn't work, visualize the cross-attentions, this could be why.

## To Dos

- [ ] Enable real video editing
- [ ] Support mode resolutions
- [ ] Intergaret DiT based T2V backbones like Open Sora















### Special thanks to the following works for open sourcing their efforts that helped us immensely in building this work

[ExponentialML/Text-To-Video-Finetuning](https://github.com/ExponentialML/Text-To-Video-Finetuning) 
<br>[silent-chen/layout-guidance](https://github.com/silent-chen/layout-guidance/tree/main)
<br>[Sainzerjj/Free-Guidance-Diffusion](https://github.com/Sainzerjj/Free-Guidance-Diffusion/tree/master)
<br>[google/prompt-to-prompt](https://github.com/google/prompt-to-prompt)

