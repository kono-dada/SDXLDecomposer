from typing import Union, Tuple, Optional
import torch
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms as tvt
from torchvision.utils import make_grid
from typing import List, Optional, Tuple, Union
from diffusers import StableDiffusionPipeline


def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


def concat_img(img_list: List[torch.Tensor] | torch.Tensor, nrow: int = 10) -> Image.Image:
    if isinstance(img_list, list):
        if img_list[0].dim() == 4:
            images = torch.cat(img_list, dim=0)
        elif img_list[0].dim() == 3:
            images = torch.stack(img_list, dim=0)
        else:
            raise ValueError(f"Unsupported dimension: {img_list[0].dim()}")
    else:
        images = img_list
    grid_image = make_grid(images, nrow=nrow)
    grid_image = grid_image.permute(1, 2, 0).cpu().numpy()
    grid_image = Image.fromarray((grid_image * 255).astype('uint8'))
    return grid_image


def pca(x: torch.Tensor, target_dim: int, baseline_V: torch.Tensor = None):  # x: (n, d)
    _x = x.float().cuda()
    _x = _x - _x.mean(dim=0)
    U, S, V = torch.svd(_x)
    V_reduced = V[:, :target_dim]
    # 如果提供了基准特征向量，则进行对齐
    if baseline_V is not None:
        for i in range(target_dim):
            dot_product = torch.dot(V_reduced[:, i], baseline_V[:, i])
            if dot_product < 0:
                V_reduced[:, i] *= -1
    projected_x = torch.mm(_x, V_reduced)
    return projected_x.cpu(), V_reduced


def interpolate_latents_and_inference(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    latent_1: torch.Tensor,
    latent_2: torch.Tensor,
    steps: int,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 20
) -> Image.Image:
    w = torch.linspace(0, 1, steps).cuda()
    # create a series of latent vectors that interpolate between latent_1 and latent_2
    # to keep the variance of the latents constant, we scale the latents by the square root of w
    latents = latent_1 * torch.sqrt(w)[:, None, None, None] + \
        latent_2 * torch.sqrt(1 - w)[:, None, None,
                                     None]  # (steps, -1, -1, -1)
    pipeline.safety_checker = None
    img = pipeline(
        prompt,
        latents=latents,
        output_type='pt',
        safety_checker=None,
        num_images_per_prompt=steps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images
    return concat_img(img.cpu())
