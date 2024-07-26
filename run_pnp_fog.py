import argparse, os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,5"
from run_features_extraction import load_model_from_config
import io
# from CLIP import clip
from torch import nn

from torchvision import transforms
from image_datasets import load_data
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange, tqdm
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import json
import logging
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.distributed as dist
import blobfile as bf
import dist_util
from dist_util import *
from ldm.util import instantiate_from_config
from torchvision.transforms import functional as TF
from ldm.models.diffusion.ddim import DDIMSampler
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from feature_exctractor import FeatureExtractorDDPM
# from DDA.image_adapt.resize_right import resize
from image_datasets import load_data
from torchvision import utils
from guided_diffusion import  logger
import torch.distributed as dist
from torchvision import transforms, models

model_config_unet = model_and_diffusion_defaults()
model_config_unet.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '50', # see sampling scheme in 4.1 (T')
    'image_size': 256,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_fp16': True,
    'use_scale_shift_norm': True,
})


normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

# # added
# def load_reference(data_dir, batch_size, image_size, class_cond=False, corruption="shot_noise", severity=5,num=1):
#     data = load_data(
#         data_dir=data_dir,
#         batch_size=batch_size,
#         image_size=image_size,
#         class_cond=class_cond,
#         deterministic=True,
#         random_flip=False,
#         corruption=corruption,
#         severity=severity,
#         num_per_class=num,
#     )
#     for large_batch, model_kwargs, filename in data:
#         model_kwargs["ref_img"] = large_batch
#         yield model_kwargs, filename

# load input image into tensor range in [-1, 1]
def load_img(path,size):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"loaded input image of size ({x}, {y}) from {path}")
    h = w = size
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

# load sd-1.4 with its default config
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model

def zecon_loss(x0_features_list, x0_t_features_list, temperature=0.07):
    loss_sum = 0
    num_layers = len(x0_features_list)
    print(f"Number of layers: {num_layers}")
    for x0_features, x0_t_features in zip(x0_features_list, x0_t_features_list):
        batch_size, feature_dim, h, w = x0_features.size()
        x0_features = x0_features.view(batch_size, feature_dim, -1)
        x0_t_features = x0_t_features.view(batch_size, feature_dim, -1)

        # Compute the similarity matrix
        sim_matrix = torch.einsum('bci,bcj->bij', x0_features, x0_t_features)
        sim_matrix = sim_matrix / temperature

        # Create positive and negative masks
        pos_mask = torch.eye(h * w, device=sim_matrix.device).unsqueeze(0).bool()
        neg_mask = ~pos_mask

        # Compute the loss using cross-entropy
        logits = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]
        labels = torch.arange(h * w, device=logits.device)
        logits_1d = logits.view(-1)[neg_mask.view(-1)]
        labels_1d = labels.repeat(batch_size * (h * w - 1)).unsqueeze(0).to(torch.float)
        layer_loss = F.cross_entropy(logits_1d.view(batch_size, -1), labels_1d, reduction='mean')

        loss_sum += layer_loss

    # Average the loss across layers
    loss = loss_sum / num_layers

    return loss

def pixel_loss(x, x_t):
    loss = nn.MSELoss()
    return loss(x, x_t)
        
def main():

    parser = argparse.ArgumentParser()
    ### model related config
    parser.add_argument('--config', default ='/home/majc/TTA/TTA/configs/pnp/pnp-real.yaml', help='the experiment setting configs')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--model_config', type=str, default='/home/majc/plug-and-play/configs/stable-diffusion/v1-inference.yaml', help='model config')
    parser.add_argument('--ckpt', type=str, default='/data/majc/sd-v1-4.ckpt', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument("--check-safety", action='store_true')

    ### image related config
    parser.add_argument('--scale', type=float, default=6.0, help='unconditional guidance scale')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--batch_size', type=int, default=1,help='')
    parser.add_argument('--image_size', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--base_samples', type=str, default='/data/ImageNet-C', help='dataset path')
    parser.add_argument('--corruption', type=str, default='gaussian_noise', help='corruption type')
    parser.add_argument('--severity', type=int, default=5, help='the corruption severity level')
    parser.add_argument('--num', type=int, default=1, help='for each calss select num images')
    parser.add_argument('--num_samples', type=int, default=1000, help='for each calss select num images')

    parser.add_argument('--save_dir', type=str, default='/home/majc/TTA_results/Ours', help='the results iamges save dir')
    parser.add_argument("--save_all_features", action="store_true", help="if set to true, saves all feature maps, otherwise only saves those necessary for PnP , the default is False")

    ### ddim related config
    parser.add_argument('--ddim_steps', type=int, default=100, help='number of ddim steps')
    parser.add_argument('--save_feature_timesteps', type=int, default=50, help='save feature timesteps')
    parser.add_argument('--prompt', type=str, default='', help='prompt text')
    parser.add_argument('--prompts', nargs='+', default=['a realistic photo'], help='text prompts for translations')
    parser.add_argument('--num_ddim_sampling_steps', type=int, default=50, help='number of ddim sampling steps')
    parser.add_argument('--feature_injection_threshold', type=int, default=40, help='feature injection threshold')
    parser.add_argument('--negative_prompt', type=str, default='a noised photo', help='negative prompt text')
    parser.add_argument('--negative_prompt_alpha', type=float, default=0, help='initial strength of negative prompting')
    parser.add_argument('--negative_prompt_schedule', type=str, default='linear', choices=['linear', 'constant', 'exp'], help='attenuation schedule of negative prompting')


    opt = parser.parse_args()
    exp_config = opt
    seed = -1
    seed_everything(seed)
    # dist_util.setup_dist()
    logger.configure(dir=opt.save_dir)

    print("loading model...")
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)

    # VGG = models.vgg19(pretrained=True).features
    # VGG.to(device)
    # Resnet = models.resnet18(pretrained=True)
    # Resnet.to(device)
    # for parameter in Resnet.parameters():
    #     parameter.requires_grad_(False)
    # Resnet.eval()

    # for parameter in VGG.parameters():
    #     parameter.requires_grad_(False)

    # def get_features(image, model, layers=None):

    #     if layers is None:
    #         layers = {'0': 'conv1_1',  
    #                 '5': 'conv2_1',  
    #                 '10': 'conv3_1', 
    #                 '19': 'conv4_1', 
    #                 '21': 'conv4_2', 
    #                 '28': 'conv5_1',
    #                 '31': 'conv5_2'
    #                 }  
    #     features = {}
    #     x = image
    #     for name, layer in model._modules.items():
    #         x = layer(x)   
    #         if name in layers:
    #             features[layers[name]] = x
        
    #     return features

    # def feature_loss(x, x_t):
    #     x_features = get_features(x, VGG)
    #     x_t_features = get_features(x_t, VGG)

    #     loss = 0
    #     loss += torch.mean((x_features['conv4_2'] - x_t_features['conv4_2']) ** 2)
    #     loss += torch.mean((x_features['conv5_2'] - x_t_features['conv5_2']) ** 2)

    #     return loss

    # def pixel_loss(x, x_t):
    #     loss = nn.MSELoss()
    #     return loss(x, x_t)

    # print("loading data...")
    # data = load_reference(
    #     exp_config.base_samples,
    #     exp_config.batch_size,
    #     image_size=exp_config.image_size,
    #     class_cond=False,
    #     corruption=exp_config.corruption,
    #     severity=exp_config.severity,
    #     num=exp_config.num,
    # )
    # print("loading data...")
    # if exp_config.corruption == '':
    #     folder_path = exp_config.base_samples
    # else:
    #     folder_path = f"{exp_config.base_samples}/{exp_config.corruption}/{exp_config.severity}"
    print("loading data...")
    if exp_config.corruption == '':
        folder_path = exp_config.base_samples

    else:
        folder_path = f"{exp_config.base_samples}/{exp_config.corruption}/{exp_config.severity}"
    num = exp_config.num
    memory_files = {}


    # print("loading clip model...")
    # clip_model, clip_preprocess = clip.load('ViT-B/16', jit=False)
    # clip_model = clip_model.eval().requires_grad_(False).to(device)
    # clip_size = clip_model.visual.input_resolution
    # print(f"clip input_resolution: {clip_size}")
    # resize_cropper = transforms.RandomResizedCrop(size=(clip_size, clip_size))
    # affine_transfomer = transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
    # perspective_transformer = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
    # patcher = transforms.Compose([
    #     resize_cropper,
    #     perspective_transformer,
    #     affine_transfomer
    # ])

    # some callback functions to save feature maps and result images during ddim sampling process
    # save a feature map to buffer memory files
    #     save the result images during each sampling step 
          
    def img_normalize(image):
        mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
        std=torch.tensor([0.229, 0.224, 0.225]).to(device)
        mean = mean.view(1,-1,1,1)
        std = std.view(1,-1,1,1)
        image = (image-mean)/std
        return image  
    
    def save_sampled_img(x, i, save_path):
        x_samples_ddim = model.decode_first_stage(x)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
        x_sample = x_image_torch[0]
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img.save(os.path.join(save_path, f"{i}.png"))

##### save feature_map in buffer
    def save_feature_map(feature_map, filename):
        buffer = io.BytesIO()
        torch.save(feature_map, buffer)
        memory_files[filename] = buffer

##### save feature_map in local
    def save_feature_map(feature_map, filename):
        buffer = io.BytesIO()
        torch.save(feature_map, buffer)
        memory_files[filename] = buffer


    # save feature maps of target blocks to buffer memory files
    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block in tqdm(blocks, desc="",disable=True): 
            if block_idx < 4:
            # if not opt.save_all_features and block_idx < 4:
                block_idx += 1
                continue
            if "ResBlock" in str(type(block[0])):
                # if opt.save_all_features or block_idx == 4:
                if block_idx == 4:
                    save_feature_map(block[0].in_layers_features, f"{feature_type}_{block_idx}_in_layers_features_time_{i}")
                    save_feature_map(block[0].out_layers_features, f"{feature_type}_{block_idx}_out_layers_features_time_{i}")
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                save_feature_map(block[1].transformer_blocks[0].attn1.k, f"{feature_type}_{block_idx}_self_attn_k_time_{i}")
                save_feature_map(block[1].transformer_blocks[0].attn1.q, f"{feature_type}_{block_idx}_self_attn_q_time_{i}")
            block_idx += 1

    # define what to do in first ddim sampling process (save target features in memory files)    
    def ddim_sampler_callback(pred_x0, xt, i):
        # print('i:',i)
        # print("save")
        save_feature_maps(unet_model.input_blocks, i, "input_block")
        save_feature_maps(unet_model.output_blocks , i, "output_block")
        # you can add more visualization here,such as save_sampled_img
            # save_sampled_img(pred_x0, i, predicted_samples_path)

    def ddim_sampler_callback_(pred_x0, xt, i):
        predicted_samples_path = os.path.join(exp_config.save_dir, "predicted_samples")
        predicted_samples_path_ = os.path.join(exp_config.save_dir, "xt_samples")

        os.makedirs(predicted_samples_path, exist_ok=True)
        os.makedirs(predicted_samples_path_, exist_ok=True)

        # print('i:',i)
        # you can add more visualization here,such as save_sampled_img
        save_sampled_img(pred_x0, i, predicted_samples_path)
        save_sampled_img(xt, i, predicted_samples_path_)


    # load target features from memory files
    def load_target_features():
        self_attn_output_block_indices = [4,5,6,7,8,9,10,11]
        out_layers_output_block_indices = [4]
        output_block_self_attn_map_injection_thresholds = [ddim_steps // 2] * len(self_attn_output_block_indices)
        feature_injection_thresholds = [exp_config.feature_injection_threshold]
        target_features = []

        # sampler.ddim_timesteps = np.arange(0, exp_config.ddim_steps)
        time_range = np.flip(sampler.ddim_timesteps)
        total_steps = sampler.ddim_timesteps.shape[0]

        iterator = tqdm(time_range, desc="loading source experiment features", total=total_steps)
        for i, t in enumerate(iterator):
            current_features = {}
            for (output_block_idx, output_block_self_attn_map_injection_threshold) in zip(self_attn_output_block_indices, output_block_self_attn_map_injection_thresholds):
                if i <= int(output_block_self_attn_map_injection_threshold):
                    
                    q_ = f"output_block_{output_block_idx}_self_attn_q_time_{t}"
                    k_ = f"output_block_{output_block_idx}_self_attn_k_time_{t}"

                    # use feature name to load the feature map from memory files
                    memory_files[q_].seek(0)  # find the corresponding feature map in memory files
                    output_q = torch.load(memory_files[q_]) # load the feature map from memory files

                    memory_files[k_].seek(0)  
                    output_k = torch.load(memory_files[k_])

                    # saved feature maps in a dict
                    current_features[f'output_block_{output_block_idx}_self_attn_q'] = output_q
                    current_features[f'output_block_{output_block_idx}_self_attn_k'] = output_k

            for (output_block_idx, feature_injection_threshold) in zip(out_layers_output_block_indices, feature_injection_thresholds):
                if i <= int(feature_injection_threshold):
                    out_ = f"output_block_{output_block_idx}_out_layers_features_time_{t}"
                    memory_files[out_].seek(0)  # 重置到缓冲区的起始位置
                    output_ = torch.load(memory_files[out_])
                    current_features[f'output_block_{output_block_idx}_out_layers'] = output_
            target_features.append(current_features)

        # return a list of a directory of feature maps for each time step
        return target_features
    

    print("creating samples...")
        # Traverse through each folder and its files
    for folder_name in sorted(os.listdir(folder_path)):
        _dir = os.path.join(folder_path, folder_name)
    ## For each class, sub-sample 'num' images
        count = 0
        for subdir, _, files in sorted(os.walk(_dir)):
            if exp_config.corruption == '':
                out_dir = os.path.join(exp_config.save_dir, subdir.split('/')[-1])
             
            else:
                out_dir = os.path.join(exp_config.save_dir, exp_config.corruption, str(exp_config.severity), subdir.split('/')[-1])
            print(out_dir)
            if os.path.exists(out_dir):
                print(f"{out_dir} already exists, skipping")
                continue

            for file in sorted(files):
                if count >= num:
                    break
                # process 'num' images for each class
                img_path = os.path.join(subdir, file)
                print(f"processing {img_path}")
                count+=1

                save_feature_timesteps = exp_config.save_feature_timesteps
                callback_timesteps_to_save = [save_feature_timesteps]
                print('saving feature maps at timesteps: ',save_feature_timesteps)
                # prompts = ["exp_config.ddim_config.prompt"]
                prompts = ['']
                precision_scope = autocast if opt.precision=="autocast" else nullcontext
                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():
                            uc = model.get_learned_conditioning([""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            z_enc = None
                            
                            # load the test image from its path
                            init_image = load_img(img_path,512).to(device)
                            init_image = init_image.to(device)

                            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))

                            # ddim forward init image 
                            ddim_inversion_steps = exp_config.ddim_steps
                            z_enc, _ = sampler.encode_ddim(init_latent, num_steps=ddim_inversion_steps, conditioning=c,unconditional_conditioning=uc,unconditional_guidance_scale=1.0)
                            print('z_enc:',z_enc.shape)
                            # save its encode latent z in buffer (for a following quicker load)
                            buffer = io.BytesIO()
                            torch.save(z_enc, buffer)
                            memory_files[z_enc] = buffer
                            # ddim first sampling and save feature maps using ddim_sampler_callback
                            samples_ddim, _ = sampler.sample(S=exp_config.ddim_steps,
                                            conditioning=c,
                                            batch_size=1,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=1.0,
                                            unconditional_conditioning=uc,
                                            eta=opt.ddim_eta,
                                            x_T=z_enc,
                                            img_callback=ddim_sampler_callback,
                                            callback_ddim_timesteps=save_feature_timesteps)

                # start the second ddim sampling process (load target features from memory files and inject them into the model)
                negative_prompt = "" if exp_config.negative_prompt is None else exp_config.negative_prompt
                ddim_steps = exp_config.num_ddim_sampling_steps
                ddim_steps = 50

                sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) 

                # read the condition pormpt user give
                batch_size = len(exp_config.prompts)
                prompts = exp_config.prompts

                # read the saved latent code z for the test image from memory files
                # memory_files[z_enc].seek(0) 
                # start_code = torch.load(memory_files[z_enc]) 
                # start_code = start_code.repeat(batch_size, 1, 1, 1)

                # t = torch.tensor([25], device=device)
                # start_code = model.q_sample(init_latent, t)
                # start_code = model.get_first_stage_encoding(model.encode_first_stage(start_code))
                # init_image_tensor = Image.open(img_path).convert('RGB')
                # init_image_tensor = TF.to_tensor(init_image_tensor).to(device).unsqueeze(0).mul(2).sub(1)

                # D = 8
                # img_shape = [1, 3, 512, 512]
                # shape_u = (img_shape[0], 3, img_shape[2], img_shape[3])
                # shape_d = (img_shape[0], 3, int(img_shape[2] / D), int(img_shape[3] / D))

                '''score function'''
                def score_corrector(x, y=None):
                    x_in = model.decode_first_stage(x)
                    # pred_x0_ = model.decode_first_stage(pred_x0)

                    '''adjust  func'''
                    # feat_loss = feature_loss(img_normalize(init_image), img_normalize(x_in))
                    mse_loss = pixel_loss(init_image, x_in)
                    # classifier_loss = classifier_loss(x_in, init_image)
                    # difference = resize(resize(pred_x0_, scale_factors=1.0/D, out_shape=shape_d), scale_factors=D, out_shape=shape_u) - resize(resize(init_image, scale_factors=1.0/D, out_shape=shape_d), scale_factors=D, out_shape=shape_u)
                    loss = 2 * mse_loss
                    # loss = th.linalg.norm( 10*feat_loss +100 *mse_loss)
                    return -torch.autograd.grad(loss, x)[0]

                precision_scope = autocast if opt.precision=="autocast" else nullcontext
                injected_features = load_target_features()
                unconditional_prompt = ""

                # start the second ddim sampling process
                # with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        # uc = None
                        nc = None
                        if exp_config.scale != 1.0:
                            # uc = model.get_learned_conditioning(batch_size * [unconditional_prompt])
                            nc = model.get_learned_conditioning(batch_size * [negative_prompt])
                        if not isinstance(prompts, list):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        start_code = sampler.stochastic_encode(init_latent, torch.tensor([50]).to(device))
                        
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        # samples = sampler.decode(start_code, c, 50, unconditional_guidance_scale=opt.scale,
                                                #  unconditional_conditioning=uc, )

                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                        conditioning=c,
                                                        negative_conditioning=nc,
                                                        batch_size=len(prompts),
                                                        shape=shape,
                                                        unconditional_guidance_scale=exp_config.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        score_corrector=None,
                                                        injected_features=injected_features,
                                                        negative_prompt_alpha=exp_config.negative_prompt_alpha,
                                                        img_callback=ddim_sampler_callback_,
                                                        negative_prompt_schedule=exp_config.negative_prompt_schedule,scale=1.0,
                                                        )
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).detach().numpy()
                        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                        # save the generated image
                        x_sample = x_image_torch[0]
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))

       
                        path = os.path.join(exp_config.save_dir, exp_config.corruption, str(exp_config.severity),  subdir.split('/')[-1])
                        os.makedirs(path, exist_ok=True)
                        out_path = os.path.join(path,  img_path.split('/')[-1])
                        img.save(out_path)
                        print(f"saving {img_path} to {out_path}")
                        # logger.log(f"created {count * exp_config.batch_size} samples")

        # dist.barrier()
        logger.log("sampling complete")
if __name__ == "__main__":
    main()
