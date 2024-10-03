#!/usr/bin/env python
# coding: utf-8
# %%


from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import torch.nn as nn
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable



import matplotlib.pyplot as plt

def plot_side_by_side(img_array, names=[], cmap="viridis"):
# Create a figure and display the image
    num_imgs = len(img_array)
    plt.subplots(1,num_imgs, figsize=(15,5))

    for i in range(1,num_imgs+1):
        plt.subplot(1,num_imgs,i)
        #plt.imshow(img_array[i-1].cpu().squeeze(0).permute(1,2,0).detach().numpy())
        plt.imshow(img_array[i-1],cmap=cmap)
        if len(names)>0:
            plt.title(str(names[i-1]))
        plt.axis('off')
    # Show the figure
    plt.show()
def kldiv(s_map, gt):
#     gt = torch.Tensor(gt).mean(-1).unsqueeze(-1)
#     s_map = torch.Tensor(s_map).mean(-1).unsqueeze(-1)
    s_map = s_map.squeeze(0)
    gt = gt.squeeze(0)

    print(s_map.size())
    print(gt.size())
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)

    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)
#     print(expand_gt.size())
#     print(gt.size())
    assert expand_gt.size() == gt.size()

    s_map = s_map/(expand_s_map*1.0)
    gt = gt / (expand_gt*1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)

    eps = 2.2204e-16
    result = gt * torch.log(eps + gt/(s_map + eps))
    # print(torch.log(eps + gt/(s_map + eps))   )
    res = torch.mean(torch.sum(result, 1))
    print(res)
    return res

def cc(s_map, gt):
 #   print("smap_loss", s_map.size())
 #   print("gt_loss", gt.size())
    gt = torch.Tensor(gt).mean(-1).unsqueeze(-1)
    s_map = torch.Tensor(s_map).mean(-1).unsqueeze(-1)
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    mean_gt = torch.mean(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_gt = torch.std(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    s_map = (s_map - mean_s_map) / std_s_map
    gt = (gt - mean_gt) / std_gt

    ab = torch.sum((s_map * gt).view(batch_size, -1), 1)
    aa = torch.sum((s_map * s_map).view(batch_size, -1), 1)
    bb = torch.sum((gt * gt).view(batch_size, -1), 1)

    return torch.mean(ab / (torch.sqrt(aa*bb)))


# %%


#### LOAD COCO CAPTIONS
import json
caption_file = "filename_and_captions.json"
with open(caption_file, "rt") as f:
    captions = json.load(f)
lines = captions["filenameandcaptions"]
caps = {x["file_name"]: x["text"][0].strip("\n") for x in lines}

#caps


# %%


scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False
NUM_DDIM_STEPS = 50
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model_id = "runwayml/stable-diffusion-v1-5"
#model_id = "CompVis/stable-diffusion-v1-4"
ldm_stable = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)
try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer


# %%


print(ldm_stable.unet.time_embedding)




print(ldm_stable.unet.time_proj(torch.Tensor([5])).cuda().shape)


# %%


print(ldm_stable.unet.time_embedding(ldm_stable.unet.time_proj(torch.Tensor([10])).cuda()).shape)


# ## Prompt-to-Prompt code



class LocalBlend:

    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask

    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:

            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0
        self.th=th




class EmptyControl:


    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return { "down_cross": [], "up_cross": []}
        #return {"down_cross": [], "mid_cross": [], "up_cross": [],
        #        "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.already_inverted = False

class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)

    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, lenlen):
    out = []
    ##lenlen= len(prompts)
    attention_maps = attention_store.get_average_attention()
#     print("attention_maps",attention_maps)
    is_cross = True
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
#             print("item",item.shape)
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(lenlen, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_word)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    print("FROM WHERE", from_where)
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))
    return images

def show_cross_attentio_img_only(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
#        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))
    return images
def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))


# %%





# ## Null Text Inversion code
# 

# %%


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        if "png" in image_path:
            image = np.repeat(np.expand_dims(np.array(Image.open(image_path)),axis=-1), 3, axis=-1)
        else:
            image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

class NullInversion:

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        print("next step ::: Timestep:" ,timestep, "next timestep",next_timestep )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def to_step_t(self, model_output: Union[torch.FloatTensor, np.ndarray], next_t: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = 0,next_t*self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps+1 # min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
#         print("to_step_t ::: Timestep:" ,timestep, "next timestep",next_timestep )

        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def to_step_t_batch(self, model_output: Union[torch.FloatTensor, np.ndarray], next_t: int, sample: Union[torch.FloatTensor, np.ndarray]):
#         print("next t",next_t )
        next_timesteps = next_t*(self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps)+1 # min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
#         print("to_step_t ::: Timestep:" ,timestep, "next timestep",next_timestep )
        timestep = 0
#         print("next_timesteps", next_timesteps)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timesteps]
#         print("next_timesteps shape", next_timesteps.shape)
#         print("alpha_prod_t_next shape", alpha_prod_t_next.shape)
        alpha_prod_t_next = alpha_prod_t_next.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
#         print("alpha_prod_t_next shape", alpha_prod_t_next.shape)

#         print("model_output shape", model_output.shape)
#         print("sample shape", sample.shape)

#         print("alpha_prod_t_next ", alpha_prod_t_next)
#         print("model_output ", model_output)

        beta_prod_t = 1 - alpha_prod_t
#         print("beta_prod_t shape", beta_prod_t.shape)
#         print("sample shape", sample.shape)
#         print("sample ", sample)

        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample    
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image
    
    def latent2image_tensor(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents
        image = self.model.vae.decode(latents)['sample']
        image = torch.clamp((image / 2 + 0.5),0, 1)
        return image
    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents
    
    @torch.no_grad()
    def image2latent_batch(self, images):
#         print("image2latent_batch",images.shape )
        with torch.no_grad():
#                 image = torch.from_numpy(image).float() / 127.5 - 1
#                 image = image.permute(2, 0, 1).unsqueeze(0).to(device)
#                 print( self.model.vae.encode(images))
                latents = self.model.vae.encode(images)['latent_dist'].mean
                latents = latents * 0.18215
        return latents
    
    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def init_prompt_batch(self, prompts):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
#         print("uncond_input",len(uncond_input))
#         print("uncond_input",uncond_input)

        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
#         print("uncond_embeddings",uncond_embeddings.shape)

        text_input = self.model.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt        
        
    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    @torch.no_grad()
    def forward_loop(self, image,noise):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent_n = latent.clone().detach()

        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            #noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent_n = self.next_step(noise, t, latent_n)
            all_latent.append(latent_n)
            
        return image_rec, latent,all_latent
    
    @torch.no_grad()
    def forward_to_t(self, image,t,noise):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent_n = latent.clone().detach()
#         noise = torch.randn_like(latent)

        
            #t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            #noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
        latent_n = self.to_step_t(noise, t, latent_n)
            
        return image_rec, latent,latent_n    
    
    @torch.no_grad()
    def forward_to_t_batch(self, images,t,noise):
        latent = self.image2latent_batch(images)
        latent_n = latent.clone().detach()
        latent_n = self.to_step_t_batch(noise, t, latent_n)
        return latent_n      
    
    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon,mask_constraint,initial_sal,mask_constraint_type):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        #initial_sal = initial_sal.mean()
        #initial_sal = initial_sal.repeat(1,4,1,1)
        #initial_sal = initial_sal.detach()
        #mask_constraint = mask_constraint.detach()
        #mask_constraint = mask_constraint.mean()
        plt.imshow(mask_constraint.cpu().squeeze(0).permute(1, 2, 0), cmap="gray"); plt.axis('off'); plt.show()
        mask_constraint = mask_constraint.repeat(1,4,1,1)
        negative_mask = resize_tensor_64((mask_constraint *(-1.0))+1)
        print("init sal size", initial_sal.shape, " mask size", mask_constraint.shape)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                #latents_x0 = latents_prev_rec - sigma * noise_pre

                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (latents_prev_rec - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

                current_img_pred = F.to_tensor(self.latent2image(pred_original_sample)).unsqueeze(0).to(DEVICE)
                print(current_img_pred.shape)
                print("timestep t : ", t)
                current_sal_pred = model(resize_tensor_256(current_img_pred))
                current_sal_pred = current_sal_pred / current_sal_pred.sum()
                if mask_constraint_type !=None:
                  print("Using mask constraint type: ", mask_constraint_type)
                  print("pred sal size", current_sal_pred.shape, "init sal size", initial_sal.shape, " mask size", mask_constraint.shape)

                  loss = nnf.mse_loss(negative_mask * latents_prev_rec, negative_mask * latent_prev)
                  sal_increase_loss = torch.mean(mask_constraint * (- current_sal_pred.detach() + initial_sal))
                else:

                  loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                  sal_increase_loss = torch.mean( -current_sal_pred.detach() + initial_sal)


                sal_increase_loss = 5000* sal_increase_loss #* (1-t/1000)
                print("denoising loss: ", loss.item(), " sal loss: ",sal_increase_loss.item() )
                loss = loss #+ sal_increase_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            plt.imshow(initial_sal.cpu().permute(1,2,0).detach().numpy()); plt.axis('off'); plt.show()
            plt.imshow(current_sal_pred.cpu().permute(1,2,0).detach().numpy()); plt.axis('off'); plt.show()
            plt.imshow(current_img_pred.cpu().squeeze(0).permute(1,2,0).detach().numpy()); plt.axis('off'); plt.show()
            print("mask_constraint size", mask_constraint.shape)
            plt.imshow(mask_constraint.cpu().mean(dim=1).permute(1, 2, 0), cmap="gray"); plt.axis('off'); plt.show()
            plt.imshow(negative_mask.cpu().mean(dim=1).permute(1, 2, 0), cmap="gray"); plt.axis('off'); plt.show()


            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list

    def invert_sal(self,mask_constraint_type,constraint_mask,initial_sal, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon,constraint_mask,initial_sal,mask_constraint_type)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings

    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        #uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon,constraint_mask,initial_sal,mask_constraint_type)
        return (image_gt, image_rec), ddim_latents[-1], None #uncond_embeddings

    def get_latent_z_all_t(self, image_path: str, prompt: str, noise, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
            
        image_rec, latent,ddim_latents = self.forward_loop(image_gt,noise)
        
        print("latent shape",  latent.shape)

        print("ddim latents shape", len(ddim_latents))

        print("ddim latents shape", ddim_latents[0].shape)

        return (image_gt, image_rec), latent,ddim_latents ,ddim_latents[-1], None #uncond_embeddings
    
    def get_latent_z_at_t(self, image_path: str, prompt: str, t:int, noise,offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
            
        image_rec, latent,latent_at_t = self.forward_to_t(image_gt, t,noise)
        
#         print("latent shape",  latent.shape)

        #print("forward_to_t latents shape", len(ddim_latents))

#         print("forward_to_t latent shape", latent_at_t.shape)

        return (image_gt, image_rec), latent,latent_at_t ,latent_at_t[-1], None #uncond_embeddings

    def get_latent_z_at_t_batch(self, images, prompt: str, t:int, noise, num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False, controller=None):
      #  self.init_prompt_batch(prompt)
        #ptp_utils.register_attention_control(self.model, controller)
#         image_gt = load_512(image_path, *offsets)
            
        latent_at_t = self.forward_to_t_batch(images, t,noise)
        
#         print("latent shape",  latent.shape)

        #print("forward_to_t latents shape", len(ddim_latents))

#         print("forward_to_t latent shape", latent_at_t.shape)

        return latent_at_t 
    
    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None

null_inversion = NullInversion(ldm_stable)


# ## Infernce Code

# %%


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    extra_prompt=None
):
    batch_size = len(prompt)
    if  return_type =="low": 
        ptp_utils.my_register_attention_control(model, controller)
    else:
        ptp_utils.register_attention_control(model, controller)
    height = width = 512
#     print("prompt in text2image ", prompt)
#     print("extra_prompt in text2image ", extra_prompt)   
#     print("prompt in text2image len ", len(prompt))

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

        
        
    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    #model.scheduler.set_timesteps(num_inference_steps)
    model.scheduler.set_timesteps(NUM_DDIM_STEPS)
    #early_start = 20 * num_inference_steps 
    start_time = num_inference_steps
#     print("num_inference_steps",num_inference_steps)
    controller.coeff = 1.1
    latents_old = latents
    for i, t in enumerate((model.scheduler.timesteps[-start_time:-1])):
#         print("uncond_embeddings", uncond_embeddings.shape)
#         print("DENOISING T ", t)
        controller.coeff = controller.coeff-0.1
        controller.time = t

        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
#             print("text_embeddings", text_embeddings.shape)
#             print("context", context.shape)
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
            
        if return_type =="low":
            latents = ptp_utils.diffusion_step_atn_replaced(model, controller, latents, context, t, guidance_scale, low_resource=False)

        else:
            latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
#         print("i", i,"t ", t, "earlystop", start_time)

#     if return_type == 'image':
#         image = ptp_utils.latent2image(model.vae, latents)
#     else:
#         image = latents
#     image=None
    model.scheduler.set_timesteps(NUM_DDIM_STEPS)
    return None,latents #image, latent



def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True,num_steps=1,low_resource=False, extra_prompt=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    #images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings)
    if low_resource ==True:
        images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=num_steps, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings, return_type="low",extra_prompt=extra_prompt )
    else:
        images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=num_steps, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings)

#     if verbose:
#         ptp_utils.view_images(images)
    #images=ptp_utils.latent2image(ldm_stable.vae, x_t)
    return images, x_t


# %%


class UpSample2(nn.Module):
    """
    ## Up-sampling layer
    """
    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 1, padding=0)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        # Apply convolution
        return self.conv(x)
class UpSample4(nn.Module):
    """
    ## Up-sampling layer
    """
    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 1, padding=0)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=4.0, mode="nearest")
        # Apply convolution
        return self.conv(x)
# Define the MLP module with 1x1 convs
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(512+4, 256+8, kernel_size=1, stride=1, padding=0),  
#             nn.SiLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Max-pooling to reduce spatial dimensions
            
            nn.Conv2d(256+8, 128+16, kernel_size=2, stride=2, padding=0),  
#             nn.SiLU(),       
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Max-pooling to reduce spatial dimensions
            
            nn.Conv2d(128+16, 64+32, kernel_size=1, stride=1, padding=0),  
#             nn.SiLU(),       
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Max-pooling to reduce spatial dimensions

            nn.Conv2d(64+32, 32+64, kernel_size=2, stride=2, padding=0),  
#             nn.SiLU(),       
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Max-pooling to reduce spatial dimensions
            
            nn.Conv2d(32+64, 128, kernel_size=2, stride=2, padding=0),  
#             nn.SiLU(),       
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Max-pooling to reduce spatial dimensions
            
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),  
            nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0),  
            nn.GroupNorm(num_groups=32, num_channels=256, eps=1e-6),
            nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0),  

#             nn.MaxPool2d(kernel_size=2, stride=2),  # Max-pooling to reduce spatial dimensions
#             nn.Softmax()
                )
        self.output_layer = nn.Linear(256, 256)


    def forward(self, x):
#         print("x size in mlp", x.shape)
        x = self.convs(x)
#         print("x size in mlp after convs", x.shape)
        x = x.squeeze(-1).squeeze(-1)
        x = self.output_layer(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        return x


# %%

class VectorToZ(nn.Module):
    def __init__(self):
        super(VectorToZ, self).__init__()
        self.conv_transpose1 = nn.Sequential(
#             nn.GroupNorm(num_groups=1, num_channels=90, eps=1e-6),
            nn.Conv2d(60+30, 48, kernel_size=3, stride=1, padding=0),  # 256x32x32 -> 256x32x32
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 256x32x32 -> 256x64x64
            nn.SiLU()
        )
        self.conv_transpose2 = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, stride=1, padding=0),  # 16x64x64 -> 8x64x64
            UpSample2(32),  # 256x32x32 -> 256x64x64
            nn.SiLU()
        )
        self.conv_transpose3 = nn.Sequential(
#             nn.GroupNorm(num_groups=1, num_channels=32+30+30, eps=1e-6),
            nn.Conv2d(32+30+30, 32, kernel_size=5, stride=1, padding=0),  # 8x64x64 -> 4x64x64
#             UpSample2(32),  # 256x32x32 -> 256x64x64
            nn.SiLU()
        )
        self.conv_transpose4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),  # 8x64x64 -> 4x64x64
            nn.SiLU(),
            UpSample2(16),  # 256x32x32 -> 256x64x64
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=0),  # 8x64x64 -> 4x64x64
            nn.SiLU(),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0),  # 8x64x64 -> 4x64x64
            #nn.SiLU(),
            UpSample2(4),  # 256x32x32 -> 256x64x64
            nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0),  # 8x64x64 -> 4x64x64

        )
        
        self.norm =  nn.GroupNorm(num_groups=6, num_channels=30, eps=1e-6)

        self.upsampler = nn.Upsample(scale_factor=1.5, mode='bilinear', align_corners=False)


    def forward(self, cmaps,latent_30):
        cmaps = cmaps.unsqueeze(0)
        cmaps=  self.norm(cmaps)
        latent_30 = latent_30.unsqueeze(0)
        x = torch.cat((latent_30,cmaps),dim=1)
        x = torch.cat((cmaps,x),dim=1)
        x = self.conv_transpose1(x)
        
#         print(x.shape)
        x = self.conv_transpose2(x)
        
        x = torch.cat((self.upsampler(latent_30),x),dim=1)
        x = torch.cat((self.upsampler(cmaps),x),dim=1)
        x = self.conv_transpose3(x)
        
#         print("after tr 3",x.shape)
        x = self.conv_transpose4(x)
        x = x.squeeze(0)
        return x


# %%


import torch.nn.functional as F

# Create the overall model
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size):
        super(Autoencoder, self).__init__()
        self.mlp = MLP(input_size, hidden_sizes, latent_size)
        self.deconv_module = VectorToZ()# DeconvModule(latent_size, 4)
        self.multiplier = nn.Parameter(torch.tensor(1.0), requires_grad=True)  # Initialize to 1.0
#         self.delinear =  nn.Linear(256, 256)

        self.mini_up = nn.Sequential(
            UpSample4(256),  # 256x32x32 -> 256x64x64
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),  # 256x32x32 -> 256x32x32
            nn.SiLU(),
            UpSample4(256),  # 256x32x32 -> 256x64x64
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),  # 16x64x64 -> 8x64x64
            nn.SiLU(),
            UpSample2(128),  # 256x32x32 -> 256x64x64
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),  # 8x64x64 -> 4x64x64
            nn.SiLU(),
            UpSample2(64),  # 256x32x32 -> 256x64x64
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),  # 8x64x64 -> 4x64x64
            nn.Conv2d(32, 30, kernel_size=3, stride=1, padding=0),  # 8x64x64 -> 4x64x64
#             nn.SiLU(),

        )
    def forward(self, x,inter,c_maps):
#         x = torch.clamp(x, -30.0, 20.0)

#         x = null_inversion.model.vae.post_quant_conv(x)
#         x = torch.clamp(x, -30.0, 20.0)
        x = torch.cat((x,inter),dim=1)

        x = x / self.multiplier
        latent_vector = self.mlp(x)
        #latent_vector= latent_vector.unsqueeze(-1).unsqueeze(-1)
        latent_30 = self.mini_up(latent_vector).squeeze(0)

#         print("latent_vector shape ", latent_vector.shape) #120x16x16
#         print("x stats ", x.mean(), x.min(), x.max(), x.sum())
#         print("latent_vector stats ", latent_vector.mean(), latent_vector.min(), latent_vector.max(), latent_vector.sum())
#         for c in range(0,30):
# #                 print(reconstructed_z[0,c,:,:].shape)
# #                 print(latent_t[0,c,:,:].shape)

# #                 print("C mse, ",c,criterion(reconstructed_z[0,c,:,:], latent_t[0,c,:,:]).item() )
# #                 print("C kl, ",c,klloss(torch.log(reconstructed_z1[0,c,:,:]), latent_t1[0,c,:,:]).item() )

#                 save_img(latent_30[c,:,:],"latent_30"+str(c))
#                 save_img(c_maps[c,:,:],"c_maps"+str(c))
    
#         print("latent_30 stats ", latent_30.mean(), latent_30.min(), latent_30.max(), latent_30.sum())
#         print("c_maps stats ", c_maps.mean(), c_maps.min(), c_maps.max(), c_maps.sum())

#         print("latent_30 shape ", latent_30.shape) #120x16x16
#         print("c_maps shape ", c_maps.shape) #120x16x16

#         latent_vector=latent_vector.squeeze(0).unsqueeze(1).unsqueeze(1).repeat(1,16,16)      
#         latent_vector = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)

#         print("spatial_maps shape ", spatial_maps.shape) #120x16x16

        reconstructed_z = self.deconv_module(c_maps,latent_30)
#         print("reconstructed_z shape ", reconstructed_z.shape) #120x16x16
        reconstructed_z = self.multiplier * reconstructed_z
        return latent_vector, reconstructed_z
    
# c_maps torch.Size([30, 16, 16])
# latent_t torch.Size([1, 4, 64, 64])
# x shape  torch.Size([1, 4, 64, 64])
# c shape  torch.Size([30, 16, 16])
# latent_vector shape  torch.Size([1, 256])


# %%
class LogMSELoss(nn.Module):
    def __init__(self):
        super(LogMSELoss, self).__init__()

    def forward(self, label, prediction):
        # Apply the logarithmic transformation to both input and target
        log_input = torch.log(label + 1e-5)  # Add a small constant to avoid taking the log of zero
        log_target = torch.log(prediction + 1e-5)
#         print(log_input)
#         print(log_target)
        # Calculate the mean squared difference between the logs
        loss = nn.MSELoss()(log_input, log_target)

        return loss

from torch.utils.data import Dataset, DataLoader
# Define the mean and standard deviation values for ImageNet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class MysalDataset(Dataset):

    def __init__(self, data_folder, caps, transform=None):
        self.data_folder = data_folder
        self.image_files = [f for f in os.listdir(data_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.image_files = self.image_files[:10000]
        self.transform = transform
        self.caps = caps
        self.maps_folder = "/sinergia/bahar/fimplenet/saliency/salicon/maps/train/"
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        map_name = os.path.join(self.maps_folder, self.image_files[idx].split(".")[0]+'.png')
        salmap = Image.open(map_name).convert('L')  
        salmap = transform64(salmap)
        
        if self.transform:
            image = self.transform(image)
        image_np = np.array(image)

        image = torch.from_numpy(image_np).float() / 127.5 - 1
        image = image.permute(2, 0, 1).to(device)

        only_name = img_name.split("/")[-1]
        prompt = self.caps[only_name]    
#         print(only_name)
#         print(idx,prompt)
        return (image, only_name, prompt,salmap)

def save_img(your_tensor,name):
    convert=False
    print(your_tensor.shape)
    if your_tensor.shape[0]==1:
        convert=True
    your_tensor = your_tensor - your_tensor.min()
    your_tensor = your_tensor / your_tensor.max()
    # Convert the PyTorch tensor to a NumPy array
    tensor_as_numpy = your_tensor.detach().cpu().numpy()
    
    # Scale the values to the 0-255 range (assuming they are in [0, 1])
    scaled_numpy = ( 255 *tensor_as_numpy).astype(np.uint8)
    if convert:
        image = Image.fromarray(scaled_numpy).convert('RGB').resize((256, 256))
    else:
        # Create a PIL Image from the NumPy array
        image = Image.fromarray(scaled_numpy).resize((256, 256))

    # Specify the file path where you want to save the image
    file_path = './cmaps/'+str(name)+'.png'  # You can use other image formats like .jpg or .jpeg

    # Save the image
    image.save(file_path)

# %%


from torchvision import transforms
from torchvision.datasets import ImageFolder
import os 

def write_to_file(mytext):
    # Convert the output tensor to a format that can be written to a file
    # In this example, we'll assume you want to write it as a plain text file
    output_text = str(mytext)

    # Write the output to the file
    with open(output_file_path, 'a') as f:
        f.write(output_text)

    # Optionally, you can also close the file explicitly
    f.close()



def calculate_noise_to_add( model_output, next_t):
        next_timesteps = next_t*(ldm_stable.scheduler.config.num_train_timesteps // ldm_stable.scheduler.num_inference_steps)+1 
        timestep = 0
        alpha_prod_t = ldm_stable.scheduler.alphas_cumprod[timestep] if timestep >= 0 else ldm_stable.scheduler.final_alpha_cumprod
        alpha_prod_t_next = ldm_stable.scheduler.alphas_cumprod[next_timesteps]
        alpha_prod_t_next = alpha_prod_t_next.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
        beta_prod_t = 1 - alpha_prod_t
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        nnoise = beta_prod_t ** 0.5 * model_output
        return next_sample_direction,alpha_prod_t,beta_prod_t,alpha_prod_t_next, nnoise,
    
def to_step_t_short( next_sample_direction,alpha_prod_t,beta_prod_t,alpha_prod_t_next, nnoise,sample):
        next_original_sample = (sample - nnoise) / alpha_prod_t ** 0.5
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample    

def train():
    cd_model.train()

    # Define your optimizer (e.g., Adam)
    optimizer = torch.optim.Adam(cd_model.parameters(), lr=0.01)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=500)

    # Training loop
    num_epochs = 20000
    t=10
    noise = torch.randn([batch_size, 4, 64, 64]).cuda()


    for epoch in range(num_epochs):
        j = 0
        mse_epoch_loss = 0
        kl_epoch_loss = 0
        miniloss_mse = 0
        miniloss_kl = 0

        for images, image_paths, prompts in dataloader:
            t = torch.randint(low=0, high=1, size=(batch_size,))#.cuda()#FIXIT
            images = images.cuda()
            images = Variable(images)

    #         print("image size", images.shape)
    #         print("image_paths size", len(image_paths))
    #         print("image_paths ", image_paths)
    #         print("prompts size", len(prompts))
    #         print("prompts ",prompts)

            controller = AttentionStore()

    #         print(uncond_embeddings.shape)
    #         print("prompts", prompts, len(prompts.split(" ")))
            num_steps= torch.ones_like(t)#.cuda()
            target_step =1

            with torch.no_grad():
    #             print("BEFORE LATENT T")
                latent_t = null_inversion.get_latent_z_at_t_batch(images, None, target_step, noise, verbose=True,controller=None)  
                image_inv, x_t = run_and_display(prompts, controller, run_baseline=False, latent=latent_t, uncond_embeddings=None, verbose=False,num_steps=target_step)
            select = 0
            res = 16
            from_where = ["up", "down"]
           # attention_store = controller #AttentionStore()
            if True:
            #for p in prompts:
    #             tokens = ldm_stable.tokenizer.encode([prompts][select])
                tokens = prompts[0].split(" ")
    #             print(tokens)
                with torch.no_grad():
                    attention_maps = aggregate_attention(controller, res, from_where, True, select,1) # change to batch size, len(prompts))

                    num_tokens = len(tokens)+2
                    if num_tokens >= max_words:
                        num_tokens = 29
                        print(tokens)
                    attention_maps = torch.permute(attention_maps, (2, 0, 1))
                    maps_tensor = attention_maps[:num_tokens]
    #                 print("crosatnmaps", maps_tensor.shape)
    #             for m in maps_tensor:
    #                 plot_side_by_side(m)
                num_empty_maps = max_words - num_tokens
                empty_maps =  torch.zeros((num_empty_maps,16,16)).cuda()
                c_maps = torch.cat((maps_tensor,empty_maps),dim=0)#.cuda()

            latent_vector, reconstructed_z = cd_model(latent_t,c_maps)
            reconstructed_z = reconstructed_z.unsqueeze(0)

            loss1 = criterion(reconstructed_z, latent_t)
            reconstructed_z =reconstructed_z-reconstructed_z.min()
            reconstructed_z = reconstructed_z / reconstructed_z.sum()+ 1e-8
            latent_t =latent_t -latent_t.min()        
            latent_t = latent_t/latent_t.sum() + 1e-8
            loss_kl = klloss(torch.log(reconstructed_z), latent_t)
            loss = loss_kl + loss1

            loss.backward()
            miniloss_mse += loss1.item()
            miniloss_kl += loss_kl.item()

            mse_epoch_loss += loss1.item()
            kl_epoch_loss += loss_kl.item()

            torch.nn.utils.clip_grad_norm_(cd_model.parameters(), 5.0)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if j %100 == 0 :
                print(str(round(j/num_imgs*100))+'%', "avg loss kl: ", miniloss_kl / (100) )
                print(str(round(j/num_imgs*100))+'%', "avg loss mse: ", miniloss_mse / (100) )
                miniloss_mse = 0
                miniloss_kl = 0
            j+=1
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss kl: {kl_epoch_loss / num_imgs}, Average Loss mse: {mse_epoch_loss / num_imgs}')

        write_to_file((f'Epoch [{epoch+1}/{num_epochs}], Average Loss kl: {kl_epoch_loss / num_imgs}, Average Loss mse: {mse_epoch_loss / num_imgs}\n'))
        if epoch %2 ==0:
                torch.save(cd_model, 'cd_model-4.pth')


