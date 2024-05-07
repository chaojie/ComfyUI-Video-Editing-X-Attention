---
license: cc-by-nc-4.0
pipeline_tag: text-to-video
---

The original repo is [here](https://modelscope.cn/models/damo/text-to-video-synthesis/summary). 

**We Are Hiring!** (Based in Beijing / Hangzhou, China.)

If you're looking for an exciting challenge and the opportunity to work with cutting-edge technologies in AIGC and large-scale pretraining, then we are the place for you. We are looking for talented, motivated and creative individuals to join our team. If you are interested, please send your CV to us.

EMAIL: yingya.zyy@alibaba-inc.com

This model is based on a multi-stage text-to-video generation diffusion model, which inputs a description text and returns a video that matches the text description. Only English input is supported.

## Model Description

The text-to-video generation diffusion model consists of three sub-networks: text feature extraction, text feature-to-video latent space diffusion model, and video latent space to video visual space. The overall model parameters are about 1.7 billion. Support English input. The diffusion model adopts the Unet3D structure, and realizes the function of video generation through the iterative denoising process from the pure Gaussian noise video.

**This model is meant for research purposes. Please look at the [model limitations and biases](#model-limitations-and-biases) and [misuse, malicious use and excessive use](#misuse-malicious-use-and-excessive-use) sections.**

**How to expect the model to be used and where it is applicable**

This model has a wide range of applications and can reason and generate videos based on arbitrary English text descriptions.

## How to use

 
The model has been launched on [ModelScope Studio](https://modelscope.cn/studios/damo/text-to-video-synthesis/summary) and [huggingface](https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis), you can experience it directly; you can also refer to [Colab page](https://colab.research.google.com/drive/1uW1ZqswkQ9Z9bp5Nbo5z59cAn7I0hE6R?usp=sharing#scrollTo=bSluBq99ObSk) to build it yourself.
In order to facilitate the experience of the model, users can refer to the [Aliyun Notebook Tutorial](https://modelscope.cn/headlines/detail/26) to quickly develop this Text-to-Video model.

This demo requires about 16GB CPU RAM and 16GB GPU RAM. Under the ModelScope framework, the current model can be used by calling a simple Pipeline, where the input must be in dictionary format, the legal key value is 'text', and the content is a short text. This model currently only supports inference on the GPU. Enter specific code examples as follows:


### Operating environment (Python Package)

```
pip install modelscope==1.4.2
pip install open_clip_torch
pip install pytorch-lightning
```

### Code example (Demo Code)

```python
from huggingface_hub import snapshot_download

from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import pathlib

model_dir = pathlib.Path('weights')
snapshot_download('damo-vilab/modelscope-damo-text-to-video-synthesis',
                   repo_type='model', local_dir=model_dir)

pipe = pipeline('text-to-video-synthesis', model_dir.as_posix())
test_text = {
        'text': 'A panda eating bamboo on a rock.',
    }
output_video_path = pipe(test_text,)[OutputKeys.OUTPUT_VIDEO]
print('output_video_path:', output_video_path)
```

### View results

The above code will display the save path of the output video, and the current encoding format can be played normally with [VLC player](https://www.videolan.org/vlc/).

The output mp4 file can be viewed by [VLC media player](https://www.videolan.org/vlc/). Some other media players may not view it normally.

## Model limitations and biases

* The model is trained based on public data sets such as Webvid, and the generated results may have deviations related to the distribution of training data.
* This model cannot achieve perfect film and television quality generation.
* The model cannot generate clear text.
* The model is mainly trained with English corpus and does not support other languages ​​at the moment**.
* The performance of this model needs to be improved on complex compositional generation tasks.

## Misuse, Malicious Use and Excessive Use

* The model was not trained to realistically represent people or events, so using it to generate such content is beyond the model's capabilities.
* It is prohibited to generate content that is demeaning or harmful to people or their environment, culture, religion, etc.
* Prohibited for pornographic, violent and bloody content generation.
* Prohibited for error and false information generation.

## Training data

The training data includes [LAION5B](https://huggingface.co/datasets/laion/laion2B-en), [ImageNet](https://www.image-net.org/), [Webvid](https://m-bain.github.io/webvid-dataset/) and other public datasets. Image and video filtering is performed after pre-training such as aesthetic score, watermark score, and deduplication.

## Citation

```bibtex
    @InProceedings{VideoFusion,
        author    = {Luo, Zhengxiong and Chen, Dayou and Zhang, Yingya and Huang, Yan and Wang, Liang and Shen, Yujun and Zhao, Deli and Zhou, Jingren and Tan, Tieniu},
        title     = {VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2023}
    }
```
