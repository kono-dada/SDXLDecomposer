import torch
from diffusers import DiffusionPipeline
from typing import List
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from utils import concat_img
from utils import pca


class SDHiddenRecorder:
    """
    the schema of the hidden recorder is like
    [
        {
            "timestep": 999,
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k": Tensor,
            ...
        },
        ...
    ]
    """
    def __init__(self, pipe: DiffusionPipeline, layer_ids: List[str]):
        self.pipe = pipe
        self.unet = pipe.unet
        self.hiddens = [{}]
        for name, module in self.unet.named_modules():
            if name in layer_ids:
                print(f'Hooking {name}')
                module.register_forward_hook(self.get_hook_fn(name))
                
    def get_hook_fn(self, module_name: str):
        def hook_fn(module, input, output):
            self.hiddens[-1][module_name] = output.cpu()
        return hook_fn
    
    def __call__(self, p: DiffusionPipeline, i: int, t, kwargs):
        self.hiddens[-1]["timestep"] = t
        self.hiddens.append({})
        return kwargs
    
    def qkv_visualization(self, attn_id: str, dim=3, chunk=1):
        r"""
        Visualize the QKV of the attentions by projecting them into a 3D (RGB) space.
        
        Args:
            attn_id (int): The index of the attention.
            qkv (str): The QKV to visualize. It can be 'q', 'k', 'v'.
            dim (int): The dimension of the projected space.
            chunk (int): The chunk of the QKV. If guidance_scale > 1, there will be two latents.
                The first chunk is the unconditional (negative prompt) latent, and the second chunk is the conditional latent.
        """
        img_list = []
        for attn in tqdm(self.hiddens):
            tgt = attn[attn_id]  # (heads, h*w, c)
            h, n, c = tgt.shape
            tgt = torch.chunk(tgt, 2, dim=0)[chunk]
            tgt = tgt.permute(1, 2, 0).reshape(n, -1)  # (h*w, c*heads)
            hxw = tgt.shape[-2]
            size = int(hxw ** 0.5)
            tgt = pca(tgt, dim)
            image = tgt.transpose(-1, -2).reshape(dim, size, size)  # (dim, h, w)
            image = (image - image.min()) / (image.max() - image.min())
            image = image.unsqueeze(0)
            img_list.append(image)
        plt.figure(figsize=(20, 10))
        plt.title(f'{attn_id}')
        plt.imshow(concat_img(img_list))
        
    def feature_visualization(self, feature_id: str, dim=3, chunk=1):
        r"""
        Visualize the QKV of the attentions by projecting them into a 3D (RGB) space.
        
        Args:
            attn_id (int): The index of the attention.
            qkv (str): The QKV to visualize. It can be 'q', 'k', 'v'.
            dim (int): The dimension of the projected space.
            chunk (int): The chunk of the QKV. If guidance_scale > 1, there will be two latents.
                The first chunk is the unconditional (negative prompt) latent, and the second chunk is the conditional latent.
        """
        img_list = []
        for attn in tqdm(self.hiddens):
            tgt = attn[feature_id]  # (b, c, h, w)
            b, c, h, w = tgt.shape
            tgt = tgt.reshape(b, c, -1).permute(0, 2, 1)
            h, n, c = tgt.shape
            tgt = torch.chunk(tgt, 2, dim=0)[chunk]
            tgt = tgt.permute(1, 2, 0).reshape(n, -1)  # (h*w, c*heads)
            hxw = tgt.shape[-2]
            size = int(hxw ** 0.5)
            tgt = pca(tgt, dim)
            image = tgt.transpose(-1, -2).reshape(dim, size, size)  # (dim, h, w)
            image = (image - image.min()) / (image.max() - image.min())
            image = image.unsqueeze(0)
            img_list.append(image)
        plt.figure(figsize=(20, 10))
        plt.title(f'{feature_id}')
        plt.imshow(concat_img(img_list))
        

