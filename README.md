# Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation (CVPR 2023)

## [<a href="https://pnp-diffusion.github.io/" target="_blank">Project Page</a>] [<a href="https://github.com/MichalGeyer/pnp-diffusers" target="_blank">Diffusers Implementation</a>]

[![arXiv](https://img.shields.io/badge/arXiv-PnP-b31b1b.svg)](https://arxiv.org/abs/2211.12572) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/hysts/PnP-diffusion-features) <a href="https://replicate.com/arielreplicate/plug_and_play_image_translation"><img src="https://replicate.com/arielreplicate/plug_and_play_image_translation/badge"></a> [![TI2I](https://img.shields.io/badge/benchmarks-TI2I-blue)](https://www.dropbox.com/sh/8giw0uhfekft47h/AAAF1frwakVsQocKczZZSX6La?dl=0)

![teaser](assets/teaser.png)

# Updates:

**19/06/23** 🧨 Diffusers implementation of Plug-and-Play is available [here](https://github.com/MichalGeyer/pnp-diffusers).

## TODO:



## Usage
### 1、Downloading StableDiffusion Weights

Download the StableDiffusion weights from the [CompVis organization at Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
(download the `sd-v1-4.ckpt` file)

```
mkdir models
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 
``` 

### 2 、Downloading and putting Imagenet-C in args.base_samples in line 163.

### 3 、run the DDIM reverse and inverse process and DDIM sample with prompt in run_pnp_fog.py
if you want to use the extra grediant during the last sampling,please set score_corrector=score_corrector,else set score_corrector=None in line 528. To select different corruption ,set corruption='XXX' in line 164. set num=1 to generate 1 sample every class.
```
python run_pnp_fog.py 
``` 
### 4 、run the model evaluation with the generated samples in model_adapt.py
set your generated samples path in --data_dir
```
python model_adapt.py
``` 

## Setup

This code includes three process in run_pnp_fog.py:
1.  DDIM encode in run_pnp_fog.py line 229, 
2.  DDPM save features during sampling in line 242, 
3.  DDIM sample with prompt in line 288.

configs in config/pnp, sample config in contrast cooruption as pnp-real_100_5_contrast.yaml 
important parameters:
base_samples: XXX/ImageNet-C   # path to the base samples directory
corruption: contrast           # corruption type
severity: 5                    # corruption severity  
save_dir:                      # path to save the result
scale: 5                       # unconditional guidance scale. Note that a higher value encourages deviation from the source image
num: 1                         # number of samples to generate(sub sample)  
num_samples: total samples(num * 1000) ## number of total samples to generate

To reproduce DDA baseline ,please run DDA.sh

```
bash DDA.sh

```
To reproduce Diffpure baseline ,please cd Diffpure and run bash.sh
select different number of noise adding to init image,for example please set diffpure_ratio=0.3 for 300 steps
```
cd Diffpure
bash bash.sh

```



