import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter
import random
from functools import partial
from torchvision.utils import save_image
import yaml,os

from DiffPure.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from typing import Callable, List, Optional, Union, Any, Dict
from utils import GuidedDiffusion, dict2namespace, ImageCaptioner

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        data_prefix,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.data_prefix = data_prefix
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = os.path.join(self.data_prefix, self.local_images[idx])
        
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict, self.local_images[idx]

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    corruption="",
    severity=5,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    num_per_class=1,
):

    if not data_dir:
        raise ValueError("unspecified data directory")

## dir + corruption + severity ---> dir
    if corruption == '':
        data_prefix = data_dir
    else:
        data_prefix = os.path.join(data_dir, corruption, str(severity))

    print('load test data from:', data_prefix)

    folder_to_idx = find_folders(data_prefix)
    sample = get_prefix_samples(
        data_prefix,
        num=num_per_class,
        folder_to_idx = folder_to_idx,
        extensions=["jpeg","jpg"],
        shuffle=not deterministic,
    )

    all_files = []
    classes = []
    for img_prefix, filename, gt_label in sample:
        all_files.append(filename)
        classes.append(gt_label)
    

    dataset = ImageDataset(
        image_size,
        data_prefix,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class GuidedDiffusion(torch.nn.Module):
    def __init__(self, config, t, device=None, model_dir='/data/majc/models/DDA'):
        super().__init__()
        # self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.sample_step = 1
        self.t = t

        # load model
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(self.config.model))
        model, diffusion = create_model_and_diffusion(**model_config)
        model.load_state_dict(torch.load(f'{model_dir}/256x256_diffusion_uncond.pt', map_location='cpu'))
        model.requires_grad_(False).eval().to(self.device)

        if model_config['use_fp16']:
            model.convert_to_fp16()

        self.model = model
        self.diffusion = diffusion
        self.betas = torch.from_numpy(diffusion.betas).float().to(self.device)

    def image_editing_sample(self, img, bs_id=0, tag=None):
        with torch.no_grad():
            assert isinstance(img, torch.Tensor)
            batch_size = img.shape[0]
            assert img.ndim == 4, img.ndim
            img = img.to(self.device)
            x0 = img
            xs = []
            xts = []
            for it in range(self.sample_step):
                e = torch.randn_like(x0)
                total_noise_levels = self.t
                a = (1 - self.betas).cumprod(dim=0)
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                xts.append(x.clone())
                for i in reversed(range(total_noise_levels)):
                    t = torch.tensor([i] * batch_size, device=self.device)
                    x = self.diffusion.p_sample(self.model, x, t,
                                                clip_denoised=True,
                                                denoised_fn=None,
                                                cond_fn=None,
                                                model_kwargs=None)["sample"]
                x0 = x
                xs.append(x0)
            return torch.cat(xs, dim=0), torch.cat(xts, dim=0)

class DiffPure():
    def __init__(self, steps=0.3, num=0, save_imgs=False, fname="base"):
        with open('DiffPure/configs/imagenet.yml', 'r') as f:
            config = yaml.safe_load(f)
        self.config = dict2namespace(config)
        self.runner = GuidedDiffusion(self.config, t = int(steps * int(self.config.model.timestep_respacing)), model_dir = '/data/majc/models/DDA')
        self.steps = steps
        self.save_imgs = save_imgs
        self.cnt = num
        self.fname = fname

        # if self.save_imgs:
        #     save_dir_p = f'./diffpure_images_new_key/{self.steps}/pured'
        #     if not os.path.exists(save_dir_p):
        #         os.makedirs(save_dir_p)

    def __call__(self, img):
        img_pured, img_noisy = self.runner.image_editing_sample((img.unsqueeze(0) - 0.5) * 2)
        img_pured = (img_pured.squeeze(0).to(img.dtype).to("cpu") + 1) / 2

        # if self.save_imgs:

        return img_pured
    
    def __repr__(self):
        return self.__class__.__name__ + '(steps={})'.format(self.steps)
    

def load_reference(data_dir, batch_size, image_size, corruption="shot_noise", severity=5,num=1):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        deterministic=True,
        random_flip=False,
        corruption=corruption,
        severity=severity,
        num_per_class=num,
    )
    for large_batch, model_kwargs, filename in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs, filename

def create_argparser():
    defaults = dict(
        num_samples=1000,
        batch_size=4,
        image_size=256,
        base_samples="",
        model_path="",
        save_dir="",
        corruption="",
        severity=5,
        classifier_scale=1.0,
        num=1,
        diffpure_ratio=0.3,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def image_distortion(img,i, args):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        DiffPure(steps=float(args.diffpure_ratio),num=i)
    ])
    img = transform(img)
    return img

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()

    logger.log("loading data...")
    data = load_reference(
        args.base_samples,
        args.batch_size,
        image_size=args.image_size,
        corruption=args.corruption,
        severity=args.severity,
        num=args.num,
    )

    while count * args.batch_size * dist_util.get_world_size() < args.num_samples:
        model_kwargs, filename = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        for i in range(args.batch_size):
            if args.corruption == '':
                path = os.path.join(logger.get_dir(), filename[i].split('/')[0])
                os.makedirs(path, exist_ok=True)
                out_path = os.path.join(path, filename[i].split('/')[1])
            else:
                path = os.path.join(logger.get_dir(), args.corruption, str(args.severity), filename[i].split('/')[0])
                os.makedirs(path, exist_ok=True)
                out_path = os.path.join(path, filename[i].split('/')[1])

            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

#main 函数
if __name__ == '__main__':
    main()
