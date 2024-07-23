"""SAMPLING ONLY."""
from CLIP import clip

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
from torchvision.transforms import functional as TF

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from feature_exctractor import FeatureExtractorDDPM
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
            # Load models

        # self.model_config = model_and_diffusion_defaults()
        # self.model_config.update({
        #     'attention_resolutions': '32, 16, 8',
        #     'class_cond': False,
        #     'diffusion_steps': 1000,
        #     'rescale_timesteps': True,
        #     'timestep_respacing': '50', # see sampling scheme in 4.1 (T')
        #     'image_size': 256,
        #     'learn_sigma': True,
        #     'noise_schedule': 'linear',
        #     'num_channels': 256,
        #     'num_head_channels': 64,
        #     'num_res_blocks': 2,
        #     'resblock_updown': True,
        #     'use_fp16': True,
        #     'use_scale_shift_norm': True,
        # })
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.ddpm_model, self.diffusion = create_model_and_diffusion(**self.model_config)
        # self.ddpm_model.load_state_dict(torch.load('/data/majc/models/DDA/256x256_diffusion_uncond.pt', map_location='cpu'))
        # self.ddpm_model.requires_grad_(False).eval().to(self.model.device)
        # for name, param in self.ddpm_model.named_parameters():
        #     if 'qkv' in name or 'norm' in name or 'proj' in name:
        #         param.requires_grad_()
        # if self.model_config['use_fp16']:
        #     self.ddpm_model.convert_to_fp16()

        # self.clip_model, self.clip_preprocess = clip.load('ViT-B/16', jit=False)
        # self.clip_model = self.clip_model.eval().requires_grad_(False).to(self.model.device)
        # self.clip_size = self.clip_model.visual.input_resolution
        # self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
        #                                 std=[0.26862954, 0.26130258, 0.27577711])

        # VGG = models.vgg19(pretrained=True).features
        # VGG.to(device)
        # for parameter in VGG.parameters():
        #     parameter.requires_grad_(False)
        
        # self.resize_cropper = transforms.RandomResizedCrop(size=(self.clip_size, self.clip_size))
        # self.affine_transfomer = transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
        # perspective_transformer = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
        # self.patcher = transforms.Compose([
        #     self.resize_cropper,
        #     perspective_transformer,
        #     self.affine_transfomer
        # ])


        # # Feature Exctractor
        # self.feature_extractor = FeatureExtractorDDPM(
        #     model = self.ddpm_model,
        #     blocks = [10, 11, 12, 13, 14],
        #     input_activations = False,
        #     **self.model_config
        # )
            
    # # Define loss-related functions
    # def global_loss(self,image, prompt):
    #     similarity = 1 - self.clip_model(image, prompt)[0] / 100 # clip returns the cosine similarity times 100
    #     return similarity.mean()

    # def directional_loss(self,x, x_t, p_source, p_target):
    #     encoded_image_diff = x - x_t
    #     encoded_text_diff = p_source - p_target
    #     cosine_similarity = torch.nn.functional.cosine_similarity(
    #         encoded_image_diff,
    #         encoded_text_diff,
    #         dim=-1
    #     )
    #     return (1 - cosine_similarity).mean()

    # def zecon_loss(self,x0_features_list, x0_t_features_list, temperature=0.07):
    #     loss_sum = 0
    #     num_layers = len(x0_features_list)
    #     print(f"Number of layers: {num_layers}")
    #     for x0_features, x0_t_features in zip(x0_features_list, x0_t_features_list):
    #         batch_size, feature_dim, h, w = x0_features.size()
    #         x0_features = x0_features.view(batch_size, feature_dim, -1)
    #         x0_t_features = x0_t_features.view(batch_size, feature_dim, -1)

    #         # Compute the similarity matrix
    #         sim_matrix = torch.einsum('bci,bcj->bij', x0_features, x0_t_features)
    #         sim_matrix = sim_matrix / temperature

    #         # Create positive and negative masks
    #         pos_mask = torch.eye(h * w, device=sim_matrix.device).unsqueeze(0).bool()
    #         neg_mask = ~pos_mask

    #         # Compute the loss using cross-entropy
    #         logits = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]
    #         labels = torch.arange(h * w, device=logits.device)
    #         logits_1d = logits.view(-1)[neg_mask.view(-1)]
    #         labels_1d = labels.repeat(batch_size * (h * w - 1)).unsqueeze(0).to(torch.float)
    #         layer_loss = F.cross_entropy(logits_1d.view(batch_size, -1), labels_1d, reduction='mean')

    #         loss_sum += layer_loss

    #     # Average the loss across layers
    #     loss = loss_sum / num_layers
    #     return loss

    # def get_features(self,image, model, layers=None):

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

    # def feature_loss(self,x, x_t):
    #     x_features = get_features(x, VGG)
    #     x_t_features = get_features(x_t, VGG)

    #     loss = 0
    #     loss += torch.mean((x_features['conv4_2'] - x_t_features['conv4_2']) ** 2)
    #     loss += torch.mean((x_features['conv5_2'] - x_t_features['conv5_2']) ** 2)

    #     return loss

    # def pixel_loss(self,x, x_t):
    #     loss = nn.MSELoss()
    #     return loss(x, x_t)
    
    # def img_normalize(self,image):
    #     mean=torch.tensor([0.485, 0.456, 0.406]).to(self.model.device)
    #     std=torch.tensor([0.229, 0.224, 0.225]).to(self.model.device)
    #     mean = mean.view(1,-1,1,1)
    #     std = std.view(1,-1,1,1)
    #     image = (image-mean)/std
    #     return image

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True, strength = 1.):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                    num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose, strength=strength)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def make_negative_prompt_schedule(self, negative_prompt_schedule, negative_prompt_alpha, total_steps):
        if negative_prompt_schedule == 'linear':
            negative_prompt_schedule = np.flip(np.linspace(0, 1, total_steps))
        elif negative_prompt_schedule == 'constant':
            negative_prompt_schedule = np.flip(np.ones(total_steps))
        elif negative_prompt_schedule == 'exp':
            negative_prompt_schedule = np.exp(-6 * np.linspace(0, 1, total_steps))
        else:
            raise NotImplementedError

        negative_prompt_schedule = negative_prompt_schedule * negative_prompt_alpha

        return negative_prompt_schedule


    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               negative_conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               injected_features=None,
               strength=1.,
               callback_ddim_timesteps=None,
               negative_prompt_alpha=1.0,
               negative_prompt_schedule='constant',
               img_path=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose,strength=strength)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    negative_conditioning=negative_conditioning,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    injected_features=injected_features,
                                                    callback_ddim_timesteps=callback_ddim_timesteps,
                                                    negative_prompt_alpha=negative_prompt_alpha,
                                                    negative_prompt_schedule=negative_prompt_schedule,
                                                    img_path=img_path
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape, negative_conditioning=None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      injected_features=None, callback_ddim_timesteps=None,
                      negative_prompt_alpha=1.0, negative_prompt_schedule='constant',img_path=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        callback_ddim_timesteps_list = np.flip(make_ddim_timesteps("uniform", callback_ddim_timesteps, self.ddpm_num_timesteps))\
            if callback_ddim_timesteps is not None else np.flip(self.ddim_timesteps)

        negative_prompt_alpha_schedule = self.make_negative_prompt_schedule(negative_prompt_schedule, negative_prompt_alpha, total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            injected_features_i = injected_features[i]\
                if (injected_features is not None and len(injected_features) > 0) else None
            negative_prompt_alpha_i = negative_prompt_alpha_schedule[i]
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      negative_conditioning=negative_conditioning,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      injected_features=injected_features_i,
                                      negative_prompt_alpha=negative_prompt_alpha_i
                                      )

            x, pred_x0 = outs
            if step in callback_ddim_timesteps_list:
                if callback: callback(i)
                if img_callback: img_callback(pred_x0, x, step)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(x)
                intermediates['pred_x0'].append(pred_x0)

        #     ####### load image#########
        #     init_image = Image.open(img_path).convert('RGB')
        #     init_image = init_image.resize((self.model_config['image_size'], self.model_config['image_size']), Image.LANCZOS)
        #     init_image_embedding = self.clip_preprocess(init_image).unsqueeze(0).to(self.model.device)
        #     init_image_embedding = self.clip_model.encode_image(init_image_embedding).float()
        #     init_image_tensor = TF.to_tensor(init_image).to(self.model.device).unsqueeze(0).mul(2).sub(1)

        #     ####### new added #########
        #     with torch.enable_grad():
        #         x = x.detach().requires_grad_()
        #         n = x.shape[0]
        #         cur_t = 25
        #         cutn = 32
        #         my_t = torch.ones([n], device=self.model.device, dtype=torch.long) * cur_t
        #         out = self.diffusion.p_mean_variance(self.ddpm_model, x, my_t, clip_denoised=False, model_kwargs={'y': None})
        #         fac = self.diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
        #         x_in = out['pred_xstart'] * fac + x * (1 - fac)
        #         x_in_patches = torch.cat([self.normalize(self.patcher(x_in.add(1).div(2))) for i in range(cutn)])
        #         x_in_patches_embeddings = self.clip_model.encode_image(x_in_patches).float()
        #         # g_loss = self.global_loss(x_in_patches, text_target_tokens)
        #         # dir_loss = self.directional_loss(init_image_embedding, x_in_patches_embeddings, text_embed_source, text_embed_target)
        #         # feat_loss = self.feature_loss(self.img_normalize(init_image_tensor), self.img_normalize(x_in))
        #         mse_loss = self.pixel_loss(init_image_tensor, x_in)
        #         x_t_features = self.feature_extractor.get_activations() # unet features
        #         self.ddpm_model(init_image_tensor, t)
        #         x_0_features = self.feature_extractor.get_activations() # unet features
        #         z_loss = self.zecon_loss(x_0_features, x_t_features)

        #         # loss = g_loss * 5000 + dir_loss * 5000 + feat_loss * 100 + mse_loss * 10000 + z_loss * 500
        #         loss = g_loss * 5000+ mse_loss * 100 + z_loss * 1000
        #         norm = th.linalg.norm(loss)
        #         norm_grad = torch.autograd.grad(norm, x)[0]
        #         scale = 2.0
        #         out["sample"] -= norm_grad * scale
        # return out["sample"], intermediates
        return x, intermediates


    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, negative_conditioning=None,
                      repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      injected_features=None, negative_prompt_alpha=1.0
                      ):
        b, *_, device = *x.shape, x.device

        if negative_conditioning is not None:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            uc = unconditional_conditioning
            nc = negative_conditioning

            c_in = torch.cat([nc, uc])
            e_t_negative, e_t_uncond = self.model.apply_model(x_in,
                                                     t_in,
                                                     c_in,
                                                     injected_features=injected_features
                                                     ).chunk(2)

            c_in = torch.cat([uc, c])
            e_t_uncond, e_t = self.model.apply_model(x_in,
                                                     t_in,
                                                     c_in,
                                                     injected_features=injected_features
                                                     ).chunk(2)

            e_t_tilde = negative_prompt_alpha * e_t_uncond + (1 - negative_prompt_alpha) * e_t_negative
            e_t = e_t_tilde + unconditional_guidance_scale * (e_t - e_t_tilde)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) if unconditional_guidance_scale!=1 else e_t

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)


        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def encode_ddim(self, img, num_steps,conditioning, unconditional_conditioning=None ,unconditional_guidance_scale=1.):
        
        print(f"Running DDIM inversion with {num_steps} timesteps")
        T = 999
        c = T // num_steps
        iterator = tqdm(range(0,T ,c), desc='DDIM Inversion', total= num_steps)
        steps = list(range(0,T + c,c))

        for i, t in enumerate(iterator):
            img, _ = self.reverse_ddim(img, t, t_next=steps[i+1] ,c=conditioning, unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale)

        return img, _

    @torch.no_grad()
    def reverse_ddim(self, x, t,t_next, c=None, quantize_denoised=False, unconditional_guidance_scale=1.,
                     unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
        if c is None:
            e_t = self.model.apply_model(x, t_tensor, unconditional_conditioning)
        elif unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t_tensor, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t_tensor] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod #.flip(0)
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod #.flip(0)
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[t], device=device)
        a_next = torch.full((b, 1, 1, 1), alphas[t_next], device=device) #a_next = torch.full((b, 1, 1, 1), alphas[t + 1], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[t], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_next).sqrt() * e_t
        x_next = a_next.sqrt() * pred_x0 + dir_xt
        return x_next, pred_x0   

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec