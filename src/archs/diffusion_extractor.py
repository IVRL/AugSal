from PIL import Image
import torch

from diffusers import DDIMScheduler
from archs.stable_diffusion.diffusion import (
    init_models, 
    get_tokens_embedding,
    generalized_steps,
    collect_and_resize_feats,
    collect_and_resize_attention,
    forward_to_t
)
from archs.stable_diffusion.resnet import init_resnet_func,save_last_tokens_attention,save_last_self_attention, init_attention_func
import delirdim
import numpy as np

def save_img(your_tensor,name,convert=False):
    
#     print(your_tensor.shape, your_tensor.dtype)
    
    if len(your_tensor.shape) > 2:
        your_tensor= your_tensor[0]

    if your_tensor.shape[0]==1:
        convert=True
    your_tensor = your_tensor - your_tensor.min()
    your_tensor = your_tensor / your_tensor.max()
    # Convert the PyTorch tensor to a NumPy array
    tensor_as_numpy = your_tensor.detach().cpu().numpy()
    
    # Scale the values to the 0-255 range (assuming they are in [0, 1])
    scaled_numpy = (255 *tensor_as_numpy).astype(np.uint8)

    if convert:
        image = Image.fromarray(scaled_numpy).convert('RGB').resize((256, 256))
    else:
        # Create a PIL Image from the NumPy array
        image = Image.fromarray(scaled_numpy).resize((256, 256))

    # Specify the file path where you want to save the image
    file_path = './mid_outs/'+str(name)+'.png'  # You can use other image formats like .jpg or .jpeg

    # Save the image
    image.save(file_path)









class DiffusionExtractor:
    """
    Module for running either the generation or inversion process 
    and extracting intermediate feature maps.
    """
    def __init__(self, config, device):
        self.device = device
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        self.num_timesteps = config["num_timesteps"]
        self.scheduler.set_timesteps(self.num_timesteps)
        self.generator = torch.Generator(self.device).manual_seed(config.get("seed", 0))
        self.batch_size = config.get("batch_size", 1)

        self.unet, self.vae, self.clip, self.clip_tokenizer,self.pipe = init_models(device=self.device, model_id=config["model_id"])
        self.prompt = config.get("prompt", "")
        print("self.prompt",self.prompt)
        self.negative_prompt = config.get("negative_prompt", "")
        self.change_cond(self.prompt, "cond")
        self.change_cond(self.negative_prompt, "uncond")
        
        self.diffusion_mode = config.get("diffusion_mode", "generation")
        if "idxs" in config and config["idxs"] is not None:
            self.idxs = config["idxs"]
        else:
            self.idxs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.output_resolution = config["output_resolution"]

        # Note that save_timestep is in terms of number of generation steps
        # save_timestep = 0 is noise, save_timestep = T is a clean image
        # generation saves as [0...T], inversion saves as [T...0]
        self.save_timestep = config.get("save_timestep", [])

        print(f"diffusion_mode: {self.diffusion_mode}")
        print(f"idxs: {self.idxs}")
        print(f"output_resolution: {self.output_resolution}")
        print(f"prompt: {self.prompt}")
        print(f"negative_prompt: {self.negative_prompt}")
        if config["finetune_unet"]:
            for param in self.unet.parameters():
                param.requires_grad = True  
            print("FINETUNING UNET")
        else:
            print("FROZEN UNET")
    def change_cond(self, prompt, cond_type="cond"):
        with torch.no_grad():
            with torch.autocast("cuda"):
                _, new_cond = get_tokens_embedding(self.clip_tokenizer, self.clip, self.device, prompt)
                new_cond = new_cond.expand((self.batch_size, *new_cond.shape[1:]))
                new_cond = new_cond.to(self.device)
                if cond_type == "cond":
                    self.cond = new_cond
                    self.prompt = prompt
                elif cond_type == "uncond":
                    self.uncond = new_cond
                    self.negative_prompt = prompt
                else:
                    raise NotImplementedError

    def jump_to_t(self, encoded_img, t,given_noise):                
        x0_preds,et,xs = forward_to_t(encoded_img, self.unet, self.scheduler,  t,given_noise)
        return xs
        
    def run_generation(self, latent, guidance_scale=-1, min_i=None, max_i=None):
        xs = generalized_steps(
            latent,
            self.unet, 
            self.scheduler, 
            run_inversion=False, 
            guidance_scale=guidance_scale, 
            conditional=self.cond, 
            unconditional=self.uncond, 
            min_i=min_i,
            max_i=max_i
        )
        return xs
    
    def run_inversion(self, latent, controller,guidance_scale=-1, min_i=None, max_i=None):
        xs = generalized_steps(
            latent, 
            self.unet, 
            self.scheduler, 
            controller,
            run_inversion=True, 
            guidance_scale=guidance_scale, 
            conditional=self.cond, 
            unconditional=self.uncond,
            min_i=min_i,
            max_i=max_i
        )
        return xs
    def run_inversion_val(self, latent,guidance_scale=-1, min_i=None, max_i=None):
        xs = generalized_steps(
            latent, 
            self.unet, 
            self.scheduler, 
            controller=0,
            run_inversion=True, 
            guidance_scale=guidance_scale, 
            conditional=self.cond, 
            unconditional=self.uncond,
            min_i=min_i,
            max_i=max_i
        )
        return xs
    def get_feats(self, latents, extractor_fn, preview_mode=False):
        # returns feats of shape [batch_size, num_timesteps, channels, w, h]
        if not preview_mode:
            init_resnet_func(self.unet, save_hidden=True, reset=True, idxs=self.idxs, save_timestep=self.save_timestep)
        outputs = extractor_fn(latents)

        if not preview_mode:
            feats = []
            for timestep in self.save_timestep:
                print("timestep ", timestep)
                timestep_feats = collect_and_resize_feats(self.unet, self.idxs, timestep, self.output_resolution)
                feats.append(timestep_feats)
            print("LEN FEATS ", len(feats))
            feats = torch.stack(feats, dim=1)
            
            init_resnet_func(self.unet, reset=True) #RESETS
        else:
            feats = None
        return feats, outputs
    
    def get_feats_with_attention(self, latents, controller, extractor_fn, preview_mode=False,):
        # returns feats of shape [batch_size, num_timesteps, channels, w, h]
        if not preview_mode:
            init_resnet_func(self.unet, save_hidden=True, reset=True, idxs=self.idxs, save_timestep=self.save_timestep)
#             print("HEHEHEHE")
#         print(self.cond)
        outputs, noise_pred , xs= extractor_fn(latents,controller)
#         print("outputs",len(outputs))
#         print("CONTROLLER after",controller.step_store)
#         print("extractor_fn ",extractor_fn )
        controller.already_inverted = True
        res=16
        from_where = ["up", "down"]
        select = 0
        attention_maps = delirdim.aggregate_attention(controller, res, from_where, True, select,1)
        controller.cmaps= attention_maps.unsqueeze(0)
        print("controller.cmaps.shape",controller.cmaps.shape)
        controller.map_index = 2
        controller.coeff =1        
        attention_maps = torch.permute(attention_maps, (2, 0, 1))
#         for ix,c_map in enumerate(attention_maps):
#             save_img(c_map.squeeze(0),"cmap_"+str(ix))
        if not preview_mode:
            feats = []
            attn = []
            for timestep in self.save_timestep:
                timestep_feats = collect_and_resize_feats(self.unet, self.idxs, timestep, self.output_resolution)
#                 timestep_attn = collect_and_resize_attention(self.unet, self.idxs, timestep, self.output_resolution)
                feats.append(timestep_feats)
#                 attn.append(timestep_attn)

            feats = torch.stack(feats, dim=1)
#             attn = torch.stack(attn, dim=1)     
            controller.reset()
            init_resnet_func(self.unet, reset=True)
        else:
            feats = None
        return feats, outputs, attention_maps,noise_pred,xs,controller
    def get_feats_with_attention_val(self, latents, extractor_fn, preview_mode=False,):
        # returns feats of shape [batch_size, num_timesteps, channels, w, h]
        if not preview_mode:
            init_resnet_func(self.unet, save_hidden=True, reset=True, idxs=self.idxs, save_timestep=self.save_timestep)
        print("HEHEHEHE")
        print(extractor_fn)

#         print(self.cond)
        outputs, noise_pred , xs= extractor_fn(latents,None)

        if not preview_mode:
            feats = []
            attn = []
            for timestep in self.save_timestep:
                timestep_feats = collect_and_resize_feats(self.unet, self.idxs, timestep, self.output_resolution)
                feats.append(timestep_feats)

            feats = torch.stack(feats, dim=1)
            init_resnet_func(self.unet, reset=True)
        else:
            feats = None
        return feats, outputs
    
    def latents_to_images(self, latents):
        latents = latents.to(self.device)
        latents = latents / 0.18215
        images = self.vae.decode(latents.to(self.vae.dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        return [Image.fromarray(image) for image in images]
    def latents_to_images_tensor(self, latents):
        latents = latents.to(self.device)
        latents = latents / 0.18215
        images = self.vae.decode(latents.to(self.vae.dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
#         images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
#         images = (images * 255).round().astype("uint8")
        return images
    
    
    def forward(self, images=None, latents=None, guidance_scale=-1, preview_mode=False):
#         save_last_tokens_attention(self.unet, True) 
#         save_last_self_attention(self.unet, True)   
#         set_init_atn(self.unet)
        controller = delirdim.AttentionStore()
        controller.already_inverted =False
        init_attention_func(self.unet,controller)
        print("CONTROLLER",controller)
        print("CONTROLLER",controller.step_store)
        self.change_cond(self.prompt, cond_type="cond")
        print("self.prompt ",self.prompt)
#         print("dtype ", type(images.squeeze(1).squeeze(1).squeeze(0)))
        save_img(images.squeeze(1).squeeze(1).squeeze(0),"img")

        if images is None:
            if latents is None:
                latents = torch.randn((self.batch_size, self.unet.in_channels, 512 // 8, 512 // 8), device=self.device, generator=self.generator)
            if self.diffusion_mode == "generation":
                if preview_mode:
                    extractor_fn = lambda latents: self.run_generation(latents, guidance_scale, max_i=self.end_timestep)
                else:
                    extractor_fn = lambda latents: self.run_generation(latents, guidance_scale)
            elif self.diffusion_mode == "inversion":
                raise NotImplementedError
        else:
            images = torch.nn.functional.interpolate(images, size=512, mode="bilinear")
            latents = self.vae.encode(images).latent_dist.sample(generator=None) * 0.18215

            if self.diffusion_mode == "inversion":
                extractor_fn = lambda latents,controller: self.run_inversion(latents,controller, guidance_scale)
            elif self.diffusion_mode == "generation":
                raise NotImplementedError
        
        with torch.no_grad():
            with torch.autocast("cuda"):
                print("LATENTS FROM INVERSION ",latents.shape )

#                 return self.get_feats(latents, extractor_fn, preview_mode=preview_mode)
                return self.get_feats_with_attention(latents,controller, extractor_fn, preview_mode=preview_mode)

    def val(self, images=None, latents=None, guidance_scale=-1, preview_mode=False):
#         save_last_tokens_attention(self.unet, True) 
#         save_last_self_attention(self.unet, True)   
#         set_init_atn(self.unet)

        if images is None:
            if latents is None:
                latents = torch.randn((self.batch_size, self.unet.in_channels, 512 // 8, 512 // 8), device=self.device, generator=self.generator)
            if self.diffusion_mode == "generation":
                if preview_mode:
                    extractor_fn = lambda latents: self.run_generation(latents, guidance_scale, max_i=self.end_timestep)
                else:
                    extractor_fn = lambda latents: self.run_generation(latents, guidance_scale)
            elif self.diffusion_mode == "inversion":
                raise NotImplementedError
        else:
            images = torch.nn.functional.interpolate(images, size=512, mode="bilinear")
            latents = self.vae.encode(images).latent_dist.sample(generator=None) * 0.18215

            if self.diffusion_mode == "inversion":
                extractor_fn = lambda latents,controller: self.run_inversion_val(latents, guidance_scale)
            elif self.diffusion_mode == "generation":
                raise NotImplementedError
        
        with torch.no_grad():
            with torch.autocast("cuda"):
                print("LATENTS FROM INVERSION ",latents.shape )

#                 return self.get_feats(latents, extractor_fn, preview_mode=preview_mode)
                feats, outputs = self.get_feats_with_attention_val(latents, extractor_fn, preview_mode=preview_mode)
                return feats, outputs
