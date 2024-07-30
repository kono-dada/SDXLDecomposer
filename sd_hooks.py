import torch
from diffusers import DiffusionPipeline
from typing import List
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from utils import concat_img
from utils import pca
from typing import Union, Tuple, Optional


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
        self.height = 0
        self.width = 0
        self.hook_handles = []
        for name, module in self.unet.named_modules():
            if name in layer_ids:
                self.hook_handles.append(module.register_forward_hook(self.get_hook_fn(name)))
                
    def get_hook_fn(self, module_name: str):
        def hook_fn(module, input, output):
            self.hiddens[-1][module_name] = output.cpu()
        return hook_fn
    
    def __call__(self, p: DiffusionPipeline, i: int, t, kwargs):
        self.hiddens[-1]["timestep"] = t
        if t > 1.:
            self.hiddens.append({})
        else:
            for handle in self.hook_handles:
                handle.remove()
        self.height, self.width = kwargs['latents'].shape[-2:]
        return kwargs
    
        
    def feature_visualization(self, feature_id: str, dim=3, chunk=1, show=False):
        r"""
        Visualize the QKV of the attentions or hidden states by projecting them into a 3D (RGB) space.
        
        Args:
            attn_id (int): The index of the attention.
            qkv (str): The QKV to visualize. It can be 'q', 'k', 'v'.
            dim (int): The dimension of the projected space.
            chunk (int): The chunk of the QKV. If guidance_scale > 1, there will be two latents.
                The first chunk is the unconditional (negative prompt) latent, and the second chunk is the conditional latent.
        """
        img_list = []
        for i, attn in tqdm(enumerate(self.hiddens)):
            tgt = attn[feature_id]  
            if len(tgt.shape) == 4: # (b, c, h, w)
                b, c, h, w = tgt.shape
                tgt = tgt.reshape(b, c, -1).permute(0, 2, 1)
            h, w = self.height, self.width
            b, n, c = tgt.shape  # (b, h*w, c)
            tgt = torch.chunk(tgt, 2, dim=0)[chunk]
            tgt = tgt.permute(1, 2, 0).reshape(n, -1)  # (h*w, c*heads)
            hxw = tgt.shape[-2]
            scale = (hxw / h / w) ** 0.5
            if i == 0:
                tgt, baseline_V = pca(tgt, dim)
            else:
                tgt, _ = pca(tgt, dim, baseline_V)
            image = tgt.transpose(-1, -2).reshape(dim, round(h * scale), round(w * scale))  # (dim, h, w)
            image = (image - image.min()) / (image.max() - image.min())
            image = image.unsqueeze(0)
            img_list.append(image)
        img = concat_img(img_list)
        if show:
            plt.figure(figsize=(20, 10))
            plt.title(f'{feature_id}')
            plt.imshow(img)
        else:
            return img
        
        
    def last_step_cross_attn_map_visualization(self, attn_id: int, prompt: str, temperature: float = 1, show=False):
        r"""
        Visualize the attention map of the cross-attention.
        
        Args:
            attn_id (int): The index of the attention.
            qkv (str): The QKV to visualize. It can be 'q', 'k', 'v'.
        """
        q = self.hiddens[-1][f'{attn_id}.to_q']  # (b, h*w, c)
        q = torch.chunk(q, 2, dim=0)[1].squeeze(0)  # (h*w, c)
        k = self.hiddens[-1][f'{attn_id}.to_k']  # (b, 77, c)
        k = torch.chunk(k, 2, dim=0)[1].squeeze(0)  # (77, c)
        attn_map = torch.mm(q, k.T) / torch.sqrt(torch.tensor(77)) * temperature  # (h*w, 77)
        # softmax
        _attn_map = attn_map.softmax(dim=-1)
        # reshape
        hxw, c = attn_map.shape
        scale = (hxw / self.height / self.width) ** 0.5
        h, w = round(self.height * scale), round(self.width * scale)
        attn_map = _attn_map.reshape(h, w, -1).permute(2, 0, 1)
        # visualize by grayscale
        attn_map = attn_map.unsqueeze(1)
        imgs = list(torch.chunk(attn_map, 77, dim=0))
        imgs = [(img - img.min()) / (img.max() - img.min()) for img in imgs]
        image = concat_img(imgs)
        if show:
            plt.figure(figsize=(20, 10))
            plt.title(f'cross-attn-{attn_id}')
            plt.imshow(image)
            print(self.pipe.tokenizer.tokenize(prompt))
        else:
            return image, self.pipe.tokenizer.tokenize(prompt)
        
    
    @classmethod
    def joint_feature_visualization(cls, recorders: list, feature_id: str, dim=3, chunk=1, show=False):
        r"""
        Visualize the QKV of the attentions or hidden states by projecting them into a 3D (RGB) space.
        
        Args:
            attn_id (int): The index of the attention.
            qkv (str): The QKV to visualize. It can be 'q', 'k', 'v'.
            dim (int): The dimension of the projected space.
            chunk (int): The chunk of the QKV. If guidance_scale > 1, there will be two latents.
                The first chunk is the unconditional (negative prompt) latent, and the second chunk is the conditional latent.
        """
        img_list = []
        h, w = recorders[0].height, recorders[0].width
        n_recorders = len(recorders)
        timesteps = len(recorders[0].hiddens)
        for i in tqdm(range(timesteps)):
            tgts = [recorder.hiddens[i][feature_id] for recorder in recorders]
            if len(tgts[0].shape) == 4:  # [(b, c, h, w), ...]
                tgts = [tgt.reshape(tgt.shape[0], tgt.shape[1], -1).permute(0, 2, 1) for tgt in tgts]  # [(b, h*w, c), ...]
            b, n, c = tgts[0].shape  # (b, h*w, c)
            tgts = [torch.chunk(tgt, 2, dim=0)[chunk].permute(1, 2, 0).reshape(n, -1) for tgt in tgts]  # [(h*w, c*heads), ...]
            hxw = tgts[0].shape[-2]
            scale = (hxw / h / w) ** 0.5
            # concat tgts before PCA. Obtain (n_recorders * h*w, c*heads)
            tgt = torch.cat(tgts, dim=0)
            if i == 0:
                tgt, baseline_V = pca(tgt, dim)
            else:
                tgt, _ = pca(tgt, dim, baseline_V)
            # split tgt into n_recorders pieces
            tgts = torch.chunk(tgt, n_recorders, dim=0)
            images = []
            for tgt in tgts:
                image = tgt.transpose(-1, -2).reshape(dim, round(h * scale), round(w * scale))  # (dim, h, w)
                image = (image - image.min()) / (image.max() - image.min())
                image = image.unsqueeze(0)
                images.append(image)
            # concat the image tensors to a row of images
            img = torch.cat(images, dim=2)
            img_list.append(img)
        img = torch.cat(img_list, dim=3)
        img = concat_img([img], nrow=1)
        if show:
            # plt.figure(figsize=(10, 30))
            plt.title(f'{feature_id}')
            plt.imshow(img)
        else:
            return img