"""
Function override for Huggingface implementation of latent diffusion models
to cache features. Design pattern inspired by open source implementation 
of Cross Attention Control.
https://github.com/bloc97/CrossAttentionControl
"""
import torch
import types
import torch.nn.functional as F

def init_resnet_func(
  unet,
  save_hidden=False,
  use_hidden=False,
  reset=True,
  save_timestep=[],
  idxs=[(1, 0)]
):
  def new_forward(self, input_tensor, temb):
    # https://github.com/huggingface/diffusers/blob/ad9d7ce4763f8fb2a9e620bff017830c26086c36/src/diffusers/models/resnet.py#L372
    hidden_states = input_tensor

    hidden_states = self.norm1(hidden_states)
    hidden_states = self.nonlinearity(hidden_states)

    if self.upsample is not None:
      input_tensor = self.upsample(input_tensor)
      hidden_states = self.upsample(hidden_states)
    elif self.downsample is not None:
      input_tensor = self.downsample(input_tensor)
      hidden_states = self.downsample(hidden_states)

    hidden_states = self.conv1(hidden_states)

    if temb is not None:
      temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
      hidden_states = hidden_states + temb

    hidden_states = self.norm2(hidden_states)
    hidden_states = self.nonlinearity(hidden_states)

    hidden_states = self.dropout(hidden_states)
    hidden_states = self.conv2(hidden_states)

    if self.conv_shortcut is not None:
      input_tensor = self.conv_shortcut(input_tensor)

    if save_hidden:
      if save_timestep is None or self.timestep in save_timestep:
        self.feats[self.timestep] = hidden_states
    elif use_hidden:
      hidden_states = self.feats[self.timestep]
    output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
    return output_tensor
  
  layers = collect_layers(unet, idxs)
  for module in layers:
    module.forward = new_forward.__get__(module, type(module))
    if reset:
      module.feats = {}
      module.timestep = None

def set_timestep(unet, timestep=None):
  for name, module in unet.named_modules():
    module_name = type(module).__name__
    module.timestep = timestep

def collect_layers(unet, idxs=None):
  layers = []
  for i, up_block in enumerate(unet.up_blocks):
    for j, module in enumerate(up_block.resnets):
      if idxs is None or (i, j) in idxs:
        layers.append(module)
  return layers

def collect_layers_a(unet, idxs=None):
  layers = []
  for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            layers.append(module)
  return layers

def collect_dims(unet, idxs=None):
  dims = []
  for i, up_block in enumerate(unet.up_blocks):
      for j, module in enumerate(up_block.resnets):
          if idxs is None or (i, j) in idxs:
            dims.append(module.time_emb_proj.out_features)
  return dims

def collect_feats(unet, idxs):
  feats = []
  layers = collect_layers(unet, idxs)
  for module in layers:
    feats.append(module.feats)
  return feats

def set_feats(unet, feats, idxs):
  layers = collect_layers(unet, idxs)
  for i, module in enumerate(layers):
    module.feats = feats[i]
  

# def init_attention_func(unet):
#     def new_attention(self, query, key, value, sequence_length, dim):
#         batch_size_attention = query.shape[0]
#         hidden_states = torch.zeros(
#             (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
#         )
#         slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
#         for i in range(hidden_states.shape[0] // slice_size):
#             start_idx = i * slice_size
#             end_idx = (i + 1) * slice_size
#             attn_slice = (
#                 torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx]) * self.scale
#             )
#             attn_slice = attn_slice.softmax(dim=-1)
            
#             if self.use_last_attn_slice:
#                 if self.last_attn_slice_mask is not None:
#                     new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
#                     attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
#                 else:
#                     attn_slice = self.last_attn_slice
                
#                 self.use_last_attn_slice = False
                    
#             if self.save_last_attn_slice:
#                 self.last_attn_slice = attn_slice
#                 self.save_last_attn_slice = False
                
#             if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
#                 attn_slice = attn_slice * self.last_attn_slice_weights
#                 self.use_last_attn_weights = False

#             attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])

#             hidden_states[start_idx:end_idx] = attn_slice

#         # reshape hidden_states
#         hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
#         return hidden_states

#     for name, module in unet.named_modules():
#         module_name = type(module).__name__
#         if module_name == "CrossAttention":
#             module.last_attn_slice = None
#             module.use_last_attn_slice = False
#             module.use_last_attn_weights = False
#             module.save_last_attn_slice = False
#             module._attention = new_attention.__get__(module, type(module))
            
# def set_init_atn(unet):
#     for name, module in unet.named_modules():
#             module_name = type(module).__name__
#             if module_name == "CrossAttention":
#                 print("inside aaa", module_name)
#                 init_attention_func(module)
            
            
            
            
def use_last_tokens_attention(unet,use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_slice = use
            
def use_last_tokens_attention_weights(unet,use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_weights = use
            
def use_last_self_attention(unet,use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.use_last_attn_slice = use
            
def save_last_tokens_attention(unet,save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
#         print("save_last_tokens_attention ",module_name)
        
        if module_name == "CrossAttention" and "attn2" in name:
            print("insidee attn2")
            module.save_last_attn_slice = save
            
def save_last_self_attention(unet,save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.save_last_attn_slice = save   
    
    
    
    
    
def collect_attention_layers(unet, idxs=None):
    layers = []
    print(idxs)
    #last_attn_slice
    for i, up_block in enumerate(unet.up_blocks):
        module_name = type(up_block).__name__
        if "CrossAttnUpBlock2D" in module_name:
#             print( module_name)
            for j, module in enumerate(up_block.attentions):
                for k, attention_module in enumerate(module.transformer_blocks):
#                   print(k,attention_module)       
#                   if idxs is None or (i, j) in idxs:
                    #layers.append(attention_module.attn1)
                    layers.append(attention_module.attn2)
                    print(attention_module.attn2)
                    print(attention_module.attn2.atnfeats)

#                     print(attention_module.attn2._attention)

    print("layers len",len(layers))
    return layers


def collect_feats_attention(unet, idxs):
  feats = []
  layers = collect_attention_layers(unet)
#   print("unetatnfeat", unet.atnfeats)
  for module in layers:
    print(module.timestep)
    print(module.atnfeats)

    print(module.last_attn_slice)
    feats.append(module.last_attn_slice)
  return feats


def init_attention_func(unet,controller):
    def my_ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            mask=attention_mask
            context = encoder_hidden_states
            batch_size, sequence_length, dim = x.shape
#             print("x shape in attention", x.shape)
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
#             print(place_in_unet)
#             print(attn.shape)
#             if attn.shape==torch.Size([16, 256, 77]):
            if controller.already_inverted and  attn.shape[-1]== 77: #cross attention
                target_res= attn.shape[1]
#                 print("TARGET RES",target_res )
#                 print(controller.cmaps.shape)
                cmaps=None
                if target_res == 256:
                    output_size = (16, 16)  # Desired output size
                    upscaled_tensor = F.interpolate(controller.cmaps, size=output_size, mode='bilinear', align_corners=False)            
                    cmaps = upscaled_tensor.permute(0,2,3,1).reshape(-1, 256, 30)
#                 elif target_res == 64: #32
                    
#                     output_size = (8, 8)  # Desired output size
#                     upscaled_tensor = F.interpolate(controller.cmaps, size=output_size, mode='bilinear', align_corners=False)
#                     cmaps = upscaled_tensor.permute(0,2,3,1).reshape(-1, 64, 30)                    
                    
                elif target_res == 1024: #32
                    output_size = (32, 32)  # Desired output size
                    upscaled_tensor = F.interpolate(controller.cmaps, size=output_size, mode='bilinear', align_corners=False)
                    cmaps = upscaled_tensor.permute(0,2,3,1).reshape(-1, 1024, 30)
                elif target_res == 4096: #64
                    output_size = (64, 64)  # Desired output size
                    upscaled_tensor = F.interpolate(controller.cmaps, size=output_size, mode='bilinear', align_corners=False)
#                     print("upscaled_tensor",upscaled_tensor.shape)
                    cmaps = upscaled_tensor.permute(0,2,3,1).reshape(-1, 4096, 30)
#                 print("QWEWQEQW")
#                 print("cmaps in ptputils", controller.cmaps)
#                 print("cmaps index", controller.map_index)
                if target_res !=1 and target_res !=64:# and target_res !=256:
#                     print("TARGET RES", target_res)
#                     print("cmaps controller", controller.cmaps.shape)
#                     print("cmaps coef", controller.coeff)

#                     print("cmaps local", cmaps.shape)
#                     print("attn local", attn.shape)
                    if controller.time >241:
                        co = controller.coeff**6
#                         print("CO",co)
                
                        attn[:,:,controller.map_index+1]= (1-co)*attn[:,:,controller.map_index+1]+ (co)*cmaps[:,:,controller.map_index]
#                     attn = attn.softmax(dim=-1)
                        print("ATN REPLACING")
#             print("place_in_unet forward ", place_in_unet)
            if is_cross :#and "up" in place_in_unet:
                attn = controller(attn, is_cross, place_in_unet)
#             print(attn.shape)
#             print("CONTROLLER inside",controller.step_store)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)
#             print(out.shape)

#             print(out.shape)
#             print(to_out(out).shape)

            return to_out(out)

        return forward

    
    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0
            self.already_inverted=False

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
#             print("net_.__class__.__name__ ",net_.__class__.__name__)
            if "up" in place_in_unet:
#                 print("place_in_unet ",place_in_unet)
                net_.forward = my_ca_forward(net_, place_in_unet)
                return count + 1
            if "down" in place_in_unet:
                net_.forward = my_ca_forward(net_, place_in_unet)
                return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = unet.named_children()
    for net in sub_nets:
#         print("UNETUNETUNETUNETUNETUNETUNETUNETUNET",net)
        if "down" in net[0]:
#             print("net[0] ",net[0])
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count
    return controller
    
#     for name, module in unet.named_modules():
#         module_name = type(module).__name__
# #         print(module_name)
#         if module_name == "CrossAttention":
#             module.last_attn_slice = None
#             module.use_last_attn_slice = False
#             module.use_last_attn_weights = False
#             module.save_last_attn_slice = False
#             module.forward = new_forward_atn.__get__(module, type(module))
            