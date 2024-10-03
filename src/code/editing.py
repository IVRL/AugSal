import argparse
import json
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import wandb
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
from PIL import Image
import os 
# from archs.correspondence_utils import process_image
from train_hyperfeatures import get_rescale_size, load_models, save_model, log_aggregation_network, my_log_aggregation_network
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from pycocotools import mask as coco_mask
from PIL import Image, ImageDraw
from saliency_losses import *
from dataloader_clean import *
from saliency_utils import *
import time
import sys
import prompttopromptmodified 
from torchvision.utils import save_image

# Set up transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
transform64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
max_words=30
from diffusers import StableDiffusionPipeline, DDIMScheduler

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


############feature mean salicon###########
mean_salicon= torch.tensor([[0.2455, 0.2976, 0.2764, 0.2488, 0.0809, 0.4319],
        [0.2455, 0.2975, 0.2764, 0.2488, 0.0809, 0.4319],
        [0.2455, 0.2975, 0.2764, 0.2487, 0.0809, 0.4319],
        [0.2455, 0.2975, 0.2764, 0.2487, 0.0809, 0.4319],
        [0.2456, 0.2975, 0.2765, 0.2487, 0.0809, 0.4319],
        [0.2456, 0.2974, 0.2765, 0.2486, 0.0810, 0.4318]]).cuda()
############feature std salicon############
std_salicon= torch.tensor([[0.2012, 0.2410, 0.2410, 0.0392, 0.0673, 0.1997],
        [0.2012, 0.2410, 0.2410, 0.0392, 0.0673, 0.1997],
        [0.2012, 0.2410, 0.2410, 0.0392, 0.0673, 0.1997],
        [0.2012, 0.2410, 0.2410, 0.0392, 0.0673, 0.1998],
        [0.2012, 0.2410, 0.2410, 0.0392, 0.0673, 0.1998],
        [0.2012, 0.2410, 0.2410, 0.0392, 0.0673, 0.1997]]).cuda()
##########################################




def create_folder_structure(folder_name,overwrite):
    # Check if the folder already exists
    
    if os.path.exists(folder_name):
        if overwrite:
            return 0
        else:
            raise FileExistsError(f"The folder '{folder_name}' already exists.")

    # Create the main folder
    os.makedirs(folder_name)

    # Create subfolders
    subfolders = ['images', 'saliency_gt', 'selected']
    for subfolder in subfolders:
        os.makedirs(os.path.join(folder_name, subfolder))
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
    file_path = str(name)+'.png'  # You can use other image formats like .jpg or .jpeg

    # Save the image
    image.save(file_path)



# Function to create a random square mask of size 64x64
def create_random_mask(size=512):
    mask = torch.zeros(size, size)
    start_x = torch.randint(0, size - 64, (1,))
    start_y = torch.randint(0, size - 64, (1,))
    mask[start_x:start_x+64, start_y:start_y+64] = 1
    # Resize the mask to match the background size
    mask_64 = TF.resize(mask.unsqueeze(0).unsqueeze(0), (64, 64))
    return mask.cuda(),mask_64.cuda()

def get_avg_rgb(image_tensor,mask_512):
    masked_image = image_tensor * mask_512
    # Calculate the average RGB values within the masked region
    average_rgb = masked_image.mean(dim=-1).mean(dim=-1)
    return average_rgb

def create_coco_masks(size,coords):
        start_x, start_y, w, h = coords
        original_width, original_height = 640.0, 480.0  # Example original dimensions
        new_width, new_height = 512.0, 512.0  # Desired new dimensions
        # Rescale bounding box coordinates
        start_x = int(start_x / original_width * new_width)
        start_y = int(start_y / original_height * new_height)
        w = int(w / original_width * new_width)
        h = int(h / original_height * new_height)
        mask = torch.zeros(size, size)
        mask[start_y:start_y+h,start_x:start_x+w ] = 1
        # Resize the mask to match the background size
        mask_64 = TF.resize(mask.unsqueeze(0).unsqueeze(0), (64, 64))
        return mask.cuda().unsqueeze(0),mask_64.cuda().squeeze(0)

class CustomCocoDataset(Dataset): #FINDS AN IMAGE FROM THE COCO DATASET WHICH IS ALSO IN THE SALICON DATASET, selects a mask
    def __init__(self, coco_dataset,saliency_folder=None, transform=None):
        self.coco_dataset = coco_dataset
        self.transform = transform
        self.saliency_folder = saliency_folder #"../fimplenet/saliency/salicon/saliency/train/"
        
    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        image, target = self.coco_dataset[idx]
#         print(target)
#         print("idx",idx)
#         print("Filename for image with ID {}: {}".format(image_id, filename))
#         print(image)
#         print(target)
        class_labels= []
        masks = []
        masks_64 = []
        i =0
        max_idx = 0
        max_area = 0
        trials =  [1 for _ in range(len(target))]
        print("len(target)",len(target))
        # Select a random object
        if len(target) ==0:
            mask,mask_64 = create_coco_masks(512,[0,0,639,479])
            class_label = 0
            coords = [0,0,511,511]
            return -1,-1,-1,-1,-1,-1
        else:
            image_id = target[0]["image_id"]
            # Get information about the image
            image_info = coco.loadImgs(image_id)[0] 
            filename = image_info['file_name']
#             if "COCO_train2014_000000145538" not in filename:
#                 return -1,-1,-1,-1,-1,-1
        # Extract filename
        #else:
            image_id = target[0]["image_id"]
            # Get information about the image
            image_info = coco.loadImgs(image_id)[0]            
            filename = image_info['file_name']
            print(filename)
            random_index = np.random.randint(len(target))
            trials[random_index] = 0
            obj = target[random_index]
            # Extract the mask for the selected object
#             print(obj)
    #         print(obj['segmentation'])
    #         print(obj['bbox'])
            mask,mask_64 = create_coco_masks(512,obj['bbox'])
            # Convert mask to binary
            # Get class label of the object
            class_label = obj['category_id']
            area = obj['area']/307200.0
            max_area = area
            coords = obj['bbox']
            
            while area < 0.07 : #filter out very small objects
#                 if len(target) <=2:
#                     mask,mask_64 = create_coco_masks(512,[0,0,639,479])
#                     class_label = 0
#                     obj['bbox'] = [0,0,511,511]
#                     print("BREAK ", i)
#                     break
                if i >= len(target)-1:
                    class_label = 0
                    mask,mask_64 = create_coco_masks(512,[0,0,639,479])
                    coords = [0,0,511,511]
                    print("BREAK ", i)
                    break                    
                    
                i+=1
                random_index = np.random.choice([index for index, value in enumerate(trials) if value == 1])
                obj = target[random_index]
                area = obj['area']/307200.0 #normalize image size
                coords = obj['bbox']
                class_label = obj['category_id']
                trials[random_index] = 0
#                 print(trials)
                if area < 0.07:
                    continue
                else:
                    mask,mask_64 = create_coco_masks(512,obj['bbox'])

            print("area", area)
           
        class_labels.append(class_label)
#             print("class label",idx, "-",class_label)
        if self.transform:
            image = self.transform(image)
        class_labels_tensor = torch.tensor(class_labels)
        if len(target) == 0:
            saliency = torch.zeros(1,1,64,64)
#             saliency = transform64(saliency)
        else:
            image_id = target[0]["image_id"]
            # Get information about the image
            image_info = coco.loadImgs(image_id)[0]
            # Save image with bounding box and mask
            gt_name= filename.split(".jpg")[0] + ".png"
    #         print("gt_name",gt_name)
            gt_path = self.saliency_folder+gt_name 
            gt = np.array(Image.open(gt_path).convert('L'))
            gt = gt.astype('float')
            gt = cv2.resize(gt, (64,64))
            if np.max(gt) > 1.0:
                gt = gt / 255.0
            saliency = torch.FloatTensor(gt)

        return image, mask, mask_64,class_labels_tensor, saliency,filename #labels_one_hot.squeeze(1)



def get_loader(config, shuffle):
    output_size, load_size = get_rescale_size(config)
    image_path = config["image_path"]
    dataset = ImagePairDataset( image_path, load_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config["batch_size"],
    )
    return dataset, dataloader

def get_average_brightness(image_tensor):
    non_zero_pixels = image_tensor[image_tensor != 0]
    # Calculate average brightness
    average_brightness = torch.mean(non_zero_pixels.float())
    return average_brightness

def get_contrast_inside(image_patch):
    # Find non-zero pixels
    non_zero_pixels = image_patch[image_patch != 0].float()
    if non_zero_pixels.numel() == 0:
        return torch.tensor(0.0)
    # Calculate the mean intensity of the non-zero pixels
    mean_intensity = torch.mean(non_zero_pixels)
    # Calculate the squared difference from the mean for each pixel
    squared_diff = (non_zero_pixels - mean_intensity) ** 2
    # Calculate the variance (average of squared differences)
    variance = torch.mean(squared_diff)
    # Contrast is the square root of variance
    contrast = torch.sqrt(variance)
#     print("contrast",contrast)
    return contrast    


def get_global_contrast(patch, image):
    # Find non-zero pixels in the patch
    patch_non_zero = patch[patch != 0]
    if patch_non_zero.numel() == 0:
        return torch.tensor(0.0).cuda()
    # Compute contrast of the patch
    patch_mean = patch_non_zero.mean()
    patch_std = patch_non_zero.std()
    patch_contrast = patch_std / (patch_mean + 1e-8)  # Adding a small epsilon to avoid division by zero
    
    # Compute contrast of the entire image
    image_mean = image.mean()
    image_std = image.std()
    image_contrast = image_std / (image_mean + 1e-8)
    
    # Compute absolute difference between patch contrast and image contrast
    global_contrast = torch.abs(patch_contrast - image_contrast)
#     print("global_contrast",global_contrast)
   
    return global_contrast


def mse_loss(pred, target):
    # TODO: Write your custom loss function
    return torch.nn.functional.mse_loss(pred, target)

ce =  torch.nn.CrossEntropyLoss()
def classification_loss(pred, label):
    label= label.squeeze(1)
#     print("label",label.shape)
#     print("pred",pred.shape)    
    pred1 = torch.argmax(pred, dim=-1)
    print("target",label)
    print("pred",pred1)
    
    return  ce(pred,label) 

def get_hyperfeats(diffusion_extractor, aggregation_network, imgs):
    with torch.inference_mode():
     with torch.autocast("cuda"):
        try:
              feats, outputs, attentions,noise_pred,xs,controller = diffusion_extractor.forward(imgs)
              b, s, l, w, h = feats.shape
#               if attentions:
#                   print("attentions",attentions.shape)
              print("feats",feats.shape)
              print("attentions",attentions.shape)

        except Exception as e:
            print("Exception",e)
#             return None,None
    diffusion_hyperfeats_high_array = []
    diffusion_hyperfeats_low_array = []
#     print("save_timestep", diffusion_extractor.save_timestep)  
    for t in diffusion_extractor.save_timestep: 
        timesteps = t # diffusion_extractor.save_timestep #TODO SAMPLE THIS TIMESTEP
        time_proj= diffusion_extractor.unet.time_proj(torch.tensor(timesteps).repeat(b).cuda())
#         print(diffusion_extractor.unet.time_proj)
#         print("time_proj shape", time_proj.shape)
        #print(diffusion_extractor.unet.time_embedding.linear_1.weight.dtype)
        temb = diffusion_extractor.unet.time_embedding(time_proj.to(dtype=torch.float16))   
        diffusion_hyperfeats_high,diffusion_hyperfeats_low  = aggregation_network(feats.float().view((b, -1, w, h)),temb)
        diffusion_hyperfeats_high_array.append(diffusion_hyperfeats_high)
        diffusion_hyperfeats_low_array.append(diffusion_hyperfeats_low)
    diffusion_hyperfeats_high_tensor = torch.stack(diffusion_hyperfeats_high_array)
    diffusion_hyperfeats_low_tensor = torch.stack(diffusion_hyperfeats_low_array)
    
    print("diffusion_hyperfeats_high_tensor",diffusion_hyperfeats_high_tensor.shape)
    return diffusion_hyperfeats_high_tensor.squeeze(1),diffusion_hyperfeats_low_tensor.squeeze(1),attentions,noise_pred,xs,controller
def get_hyperfeats_val(diffusion_extractor, aggregation_network, imgs):
    with torch.inference_mode():
     with torch.autocast("cuda"):
#         try:
              feats, outputs = diffusion_extractor.val(imgs)
              b, s, l, w, h = feats.shape
#               if attentions:
#                   print("attentions",attentions.shape)
              print("feats",feats.shape)
#         except Exception as e:
#             print("Exception",e)
#             return None,None
    diffusion_hyperfeats_high_array = []
    diffusion_hyperfeats_low_array = []
#     print("save_timestep", diffusion_extractor.save_timestep)  
    for t in diffusion_extractor.save_timestep: 
        timesteps = t # diffusion_extractor.save_timestep #TODO SAMPLE THIS TIMESTEP
        time_proj= diffusion_extractor.unet.time_proj(torch.tensor(timesteps).repeat(b).cuda())
#         print(diffusion_extractor.unet.time_proj)
#         print("time_proj shape", time_proj.shape)
        #print(diffusion_extractor.unet.time_embedding.linear_1.weight.dtype)
        temb = diffusion_extractor.unet.time_embedding(time_proj.to(dtype=torch.float16))   
        diffusion_hyperfeats_high,diffusion_hyperfeats_low  = aggregation_network(feats.float().view((b, -1, w, h)),temb)
        diffusion_hyperfeats_high_array.append(diffusion_hyperfeats_high)
        diffusion_hyperfeats_low_array.append(diffusion_hyperfeats_low)
    diffusion_hyperfeats_high_tensor = torch.stack(diffusion_hyperfeats_high_array)
    diffusion_hyperfeats_low_tensor = torch.stack(diffusion_hyperfeats_low_array)
    
    print("diffusion_hyperfeats_high_tensor",diffusion_hyperfeats_high_tensor.shape)
    return diffusion_hyperfeats_high_tensor.squeeze(1),diffusion_hyperfeats_low_tensor.squeeze(1)

def edit_attention(diffusion_extractor,attention_maps,prompt,salmap_16,filename,noise_pred,encoded_img,strength,controller,mode="brightness"):
    attention_maps = attention_maps.cuda()
    tokens = prompt.split(" ")
    prompts = [" ".join(tokens)]
    num_tokens = len(tokens)

    # Ensure num_tokens does not exceed max_words
    if num_tokens >= max_words:
        num_tokens = max_words - 1

    maps_tensor = attention_maps[1:num_tokens+1]
    num_empty_maps = max_words - num_tokens
    empty_maps = torch.zeros((num_empty_maps, 16, 16)).cuda()

    c_maps = torch.cat((maps_tensor, empty_maps), dim=0)

    # Find candidate regions (spatial attention map)
    candidates = c_maps * salmap_16
    max_val = c_maps.max()
    sum_tensor = candidates.sum(dim=(1, 2))
    max_sum_index = sum_tensor.argmax()

    selected_mask = (candidates[max_sum_index] > 0.00001).int()
    selected = c_maps[max_sum_index] * selected_mask
    selected_no_scale = selected

    # Scale selected map based on strength
    smask = (selected > selected.mean()).int()
    selected = selected * smask * (max_val / selected.max()) * strength

    # Update c_maps with the selected mask
    c_maps = torch.cat([c_maps[:max_sum_index], selected.unsqueeze(0), c_maps[max_sum_index:-1]], 0)

    # Modify tokens and generate new prompts
    to_repeat = str(tokens[max_sum_index])
    new_tokens = tokens[:max_sum_index] + [" "] + tokens[max_sum_index:]
    prompts = [" ".join(new_tokens)]
    old_prompt = [" ".join(tokens)]

    # Interpolate selected mask to 64x64
    select_64 = F.interpolate(selected.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
    select_64_mask = (select_64 > select_64.max() / 3).unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1).int()

    # Generate latent tensors
    latent_t = encoded_img
    #Approximated pseudo-inverse of the decoder matrix
    if mode == "redness": #approximation is aptimized for the R channel
        pseudo_vec = torch.tensor([2.1686387, -1.76613569, -4.04359594, -1.12024549]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 64, 64)
    else:
        pseudo_vec = torch.tensor([1.37575317, 1.32845229, -0.27514065, -0.90754053]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 64, 64)

    # Apply adjustments based on the selected mode
    if mode == "brightness":
        latent_t2 = latent_t + pseudo_vec * strength  # Brightness adjustment
    elif mode == "contrast":
        mean = latent_t[select_64_mask.bool()].mean()  # Calculate mean for contrast adjustment
        latent_t2 = strength * (latent_t - mean * pseudo_vec) + mean * pseudo_vec  # Contrast adjustment
    elif mode == "redness":
        latent_t2 = latent_t + pseudo_vec * strength  # Redness adjustment

    # Apply mask to latent_t2
    latent_t2 = latent_t2 * select_64_mask + latent_t * (1 - select_64_mask)

    # Get the final latent state and edited image
    t = 2
    latent_ts = diffusion_extractor.jump_to_t(latent_t2, t, noise_pred)[-1]

    # Update controller state
    controller.cmaps = c_maps.unsqueeze(0)
    controller.map_index = max_sum_index
    controller.coeff = 1

    # Run the diffusion model and get the result
    edited_image, latents = prompttopromptmodified.run_and_display(prompts, controller, latent=latent_ts, num_steps=t, low_resource=True, generator=diffusion_extractor.pipe, extra_prompt=old_prompt)

    # Convert latents back to images
    result_image_img = diffusion_extractor.latents_to_images_tensor(latents)
    recons_img = diffusion_extractor.latents_to_images_tensor(encoded_img)

    return result_image_img, select_64

def check_within_2std(original, edited, std):
    # Calculate the upper and lower bounds
    upper_bound = original + 2 * std
    lower_bound = original - 2 * std
    
    # Check if edited tensor is within bounds
    within_bounds = torch.all((edited >= lower_bound) & (edited <= upper_bound))
    
    return within_bounds

m = nn.Sigmoid()

def my_bce(initial_map, second_map, continuous_mask,save):
    criterion  = nn.BCELoss()
    continuous_mask = continuous_mask/continuous_mask.max()
    # Create binary mask
    print("continuous_mask ", continuous_mask.shape, continuous_mask.min(), continuous_mask.mean(),continuous_mask.max(),continuous_mask.sum())
    binary_mask = (continuous_mask > 0.5).float()
    print("binary_mask ", binary_mask.shape, binary_mask.min(), binary_mask.mean(),binary_mask.max(),binary_mask.sum())

    # Apply binary mask to initial and second maps
    masked_initial_map = initial_map * binary_mask
    masked_second_map = second_map * binary_mask
    diff = masked_second_map - masked_initial_map
    diff = torch.abs(m(diff)-0.5)
    diff2 = diff /diff.max()
  #  is_aug = is_aug.squeeze(-1).squeeze(-1)
    selected_diff = diff#[is_aug.bool()]
    selected_mask = binary_mask#[is_aug.bool()]
    random_number =0
    if save: #for visualization
        print("masked_initial_map ", masked_initial_map.shape, masked_initial_map.min(), masked_initial_map.mean(),masked_initial_map.max(),masked_initial_map.sum())
        print("masked_second_map ", masked_second_map.shape, masked_second_map.min(), masked_second_map.mean(),masked_second_map.max(),masked_second_map.sum())
        random_number = str(np.random.randint(100))
        torchvision.utils.save_image( masked_initial_map[0], "./debug/masked_initial_map_"+random_number+".png")
        torchvision.utils.save_image( masked_second_map[0], "./debug/masked_second_map_"+random_number+".png")
        torchvision.utils.save_image( continuous_mask[0], "./debug/continuous_mask_"+random_number+".png")
        torchvision.utils.save_image( binary_mask[0], "./debug/binary_mask_"+random_number+".png")
        torchvision.utils.save_image( diff[0], "./debug/diff_"+random_number+".png")
        torchvision.utils.save_image( diff2[0], "./debug/diff2_"+random_number+".png")

    if selected_diff.shape[0] == 0:
        return torch.tensor([0.0]).cuda() 

    # Calculate Binary Cross Entropy (BCE) between masked maps
    selectedinitial_map = (masked_initial_map > 0.00005).float()

    if save:
        torchvision.utils.save_image( selectedinitial_map[0], "./debug/selectedinitial_map"+random_number+".png")
    #binary_mask = binary_mask[is_aug.bool()]
    bce_loss = criterion(selected_diff, binary_mask)
    
    return bce_loss,random_number



def validate(config, diffusion_extractor, aggregation_network, val_dataloader):
    tic = time.time()
    device = config.get("device", "cuda")
    total_loss = 0.0
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()
    my_loss = 0.0
    with torch.no_grad():
        for (img, gt) in val_dataloader:
            img = img.to(device)
            gt = gt.to(device)
            diffusion_hyperfeats_high,diffusion_hyperfeats_low,attentions,noise_pred,xs,controller = get_hyperfeats(diffusion_extractor, aggregation_network, img)
            blur_map = aggregation_network.saliency_readout(diffusion_hyperfeats_high,diffusion_hyperfeats_low)
            # Blurring
#             blur_map = pred_map.cpu().squeeze(0).clone().numpy()
#             blur_map = blur(blur_map).unsqueeze(0).to(device)
            blur_map = blur_map.mean(dim=0)
            cc_loss.update(cc(blur_map, gt))    
            kldiv_loss.update(kldiv(blur_map, gt))    
            nss_loss.update(nss(blur_map, gt))    
            sim_loss.update(similarity(blur_map, gt))    

     #   print("VALIDATION LOSS:", my_loss/len(loader))
    print('[,   val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}  time:{:3f} minutes'.format( cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, (time.time()-tic)/60))
    sys.stdout.flush()
    wandb.log({'validation cc': cc_loss.avg})
    wandb.log({'validation kl': kldiv_loss.avg})
    return 0

def train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader,caps):
    device = config.get("device", "cuda")
    max_epochs = config["max_epochs"]
    batch_size=config["batch_size"]
    num_timesteps = len(diffusion_extractor.save_timestep)
    np.random.seed(0)
    step = 0
    strength = 0.0#9
    batch_features = []
    save_features_every = 1000

    for epoch in range(max_epochs):
          strength= random.uniform(0.02, 0.2)
          for batch in tqdm(train_dataloader):
                save = False
                edit_flag = random.randint(0, 1)
                optimizer.zero_grad()
                images, masks,masks_64, class_labels, saliency_gt,filename = batch
                if (images == -1).any():
                    continue
                imgs = images.to(device)
                masks = masks.to(device)
                masks_64 = masks_64.to(device)
#                 if "COCO_train2014_000000145179" not in filename[0]:
#                     continue
#                 print("masks", masks.shape)
#                 print("masks_64", masks_64.shape)
                class_labels = class_labels.type(torch.LongTensor)
                class_labels = class_labels.to(device)
                saliency_gt = saliency_gt.to(device)
                img_64 = TF.resize(imgs, (64, 64))
                sal_16 = TF.resize(saliency_gt, (16, 16))                
                prompt = caps[filename[0]]
#                 print("CAPTION:", prompt)
#                 print(masks.sum())
#                 print("class labels shape",class_labels.shape)
                #imgs = batch
                #imgs = imgs.to(device)
                diffusion_extractor.prompt = prompt
                with torch.no_grad():
                    before_diffusion_hyperfeats_high,before_diffusion_hyperfeats_low,attentions,noise_pred,xs,controller = get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
                if before_diffusion_hyperfeats_high == None :
                    continue     
                
                imaj_16 = images.to(torch.float16)  # Convert input tensor to torch.float16
                imaj_16 = imaj_16.cuda() 
#                 imaj_16 = imaj_16 / 255.0
                imaj_16 = 2. * imaj_16 - 1.
                encoded_img = diffusion_extractor.vae.encode(imaj_16).latent_dist.sample(generator=None) * 0.18215
              
                sal_high = before_diffusion_hyperfeats_high
                sal_low = before_diffusion_hyperfeats_low                    
                    
                saliency = aggregation_network.saliency_readout(sal_high,sal_low)
                saliency_before = saliency.mean(dim=0)                    
                with torch.no_grad():

                    diffusion_hyperfeats_high = before_diffusion_hyperfeats_high * masks_64
                    diffusion_hyperfeats_low = before_diffusion_hyperfeats_low * masks_64            
            
                    rs = aggregation_network.red_readout(diffusion_hyperfeats_low)
                    gs = aggregation_network.green_readout(diffusion_hyperfeats_low)
                    bs = aggregation_network.blue_readout(diffusion_hyperfeats_low)
                    cin_before = aggregation_network.contrast_in_readout(diffusion_hyperfeats_low)
                    concat_low_feature_img = torch.cat([img_64.repeat(diffusion_hyperfeats_low.shape[0],1,1,1),diffusion_hyperfeats_low],dim=1)
                    cout_before = aggregation_network.contrast_global_readout(concat_low_feature_img)
                    brightness_before = aggregation_network.brightness_readout(diffusion_hyperfeats_low)                
                    readout_original = torch.stack((rs, gs, bs, cin_before,cout_before,brightness_before), dim=1).squeeze(-1)
                    batch_features.append(readout_original)
                    color_pred_before = torch.stack((rs, gs, bs), dim=1).squeeze(-1)
                if edit_flag:
                    bad_edit = 1
                    #save_img(images.squeeze(0),str(filename[0])+"_"+str(epoch)+"_"+str(0))
                    while bad_edit:
                        edited_images,continuous_mask= edit_attention(diffusion_extractor,attentions, prompt, sal_16,filename[0],noise_pred,encoded_img,strength,controller) 
                        print("edited_images",edited_images.shape)
                        with torch.no_grad():
                            after_diffusion_hyperfeats_high,after_diffusion_hyperfeats_low,attentions,noise_pred,xs,controller = get_hyperfeats(diffusion_extractor, aggregation_network, edited_images)  
                            print("extraction done")        
                            #EXTRACT HYPERFEATURES BEFORE AND AFTER EDITING
                            #constrain them

                            after_diffusion_hyperfeats_high_masked = after_diffusion_hyperfeats_high * masks_64
                            after_diffusion_hyperfeats_low_masked = after_diffusion_hyperfeats_low * masks_64     
                            with torch.no_grad():
                                rs = aggregation_network.red_readout(after_diffusion_hyperfeats_low_masked)
                                gs = aggregation_network.green_readout(after_diffusion_hyperfeats_low_masked)
                                bs = aggregation_network.blue_readout(after_diffusion_hyperfeats_low_masked)               
        #                         class_pred = aggregation_network.class_readout(after_diffusion_hyperfeats_high)
                                cin = aggregation_network.contrast_in_readout(after_diffusion_hyperfeats_low_masked)
                                concat_low_feature_img = torch.cat([img_64.repeat(after_diffusion_hyperfeats_low_masked.shape[0],1,1,1),after_diffusion_hyperfeats_low_masked],dim=1)
                                cout = aggregation_network.contrast_global_readout(concat_low_feature_img)
                                brightness = aggregation_network.brightness_readout(after_diffusion_hyperfeats_low_masked)            
                                readout_edited = torch.stack((rs, gs, bs, cin,cout,brightness), dim=1).squeeze(-1)

                            print("readout_edited", readout_edited)
                            print("readout_original", readout_original)

                            # Check if edited tensor is within 2 std of the original tensor
                            if not check_within_2std(readout_original, readout_edited, std_salicon):
                                # Modify parameters or perform operations
                                # For example:
                                print("Edited tensor is not within 2 standard deviations of the original tensor. Edit is out of bounds!")
                                strength = strength*0.9
                                print("Reducing strength, New strength: ",strength )
                                # Modify parameters or perform operations here
                            else:
                                print("EDIT SUCCESSFUL")
                                bad_edit = 0
                    saliency_after = aggregation_network.saliency_readout(after_diffusion_hyperfeats_high,after_diffusion_hyperfeats_low)
                    saliency_after= saliency_after.mean(dim=0)    
                    if step%50==0:
                        save=True
                    bce_loss,image_num = my_bce(saliency_before, saliency_after, continuous_mask.unsqueeze(0),save)

                else:
                    saliency_after= saliency_before   
                    bce_loss,image_num = torch.FloatTensor([0.0]).cuda(),"0" #my_bce(saliency_before, saliency_after, continuous_mask.unsqueeze(0))
                with torch.no_grad():
                    class_pred = aggregation_network.class_readout(diffusion_hyperfeats_high)
#                 class_pred = class_pred.mean(dim=0).unsqueeze(0)
                class_labels = class_labels.repeat(num_timesteps,1)
                class_labels=class_labels.to(torch.int64)
#                 print("class_pred",class_pred.shape)
                with torch.no_grad():
                    cin = aggregation_network.contrast_in_readout(diffusion_hyperfeats_low)
#                     print("img_64", img_64.shape)
#                     print("diffusion_hyperfeats_low", diffusion_hyperfeats_low.shape)    
                    concat_low_feature_img = torch.cat([img_64.repeat(diffusion_hyperfeats_low.shape[0],1,1,1),diffusion_hyperfeats_low],dim=1)
                
                    cout = aggregation_network.contrast_global_readout(concat_low_feature_img)
                    brightness = aggregation_network.brightness_readout(diffusion_hyperfeats_low)

                    with torch.no_grad():
                        masked_img = imgs * masks
                        cin_gt = get_contrast_inside(masked_img)
                        cout_gt = get_global_contrast(masked_img,imgs)
                        br_gt = get_average_brightness(masked_img)
#                 print("saliency before", saliency_before.shape)
#                 print("saliency_after ", saliency_after.shape)
#                 print("saliency_gt ", saliency_gt.shape)
#                 print("continuous_mask ", continuous_mask.shape)
                avg_rgb_batch = get_avg_rgb(imgs,masks)
                kl_loss = kldiv(saliency_before, saliency_gt)
                cc_loss = cc(saliency_before, saliency_gt)                                     

                saliency_loss = kl_loss - cc_loss
                c_loss = classification_loss(class_pred,class_labels )
                rgb_loss = mse_loss(color_pred_before, avg_rgb_batch)
                cin_loss = mse_loss(cin_before, cin_gt)
                cout_loss = mse_loss(cout_before, cout_gt)
                brightness_loss = mse_loss(brightness_before, br_gt)
                
                loss = 0.5 * c_loss  + 0.2 * saliency_loss + 0.1 *rgb_loss+ 0.1 *cin_loss+ 0.1 *cout_loss+ 0.1 *brightness_loss + 0.1*bce_loss
                
                loss.backward()
                optimizer.step()
                if save:
                    torchvision.utils.save_image( saliency_before, "./debug/saliency_before_"+image_num+".png")
                    torchvision.utils.save_image( saliency_after, "./debug/saliency_after_"+image_num+".png")

                wandb.log({"train/c_loss": c_loss.item()}, step=step)
                wandb.log({"train/rgb_loss": rgb_loss.item()}, step=step)
                wandb.log({"train/kl_loss": kl_loss.item()}, step=step)
                wandb.log({"train/cc_loss": cc_loss.item()}, step=step)
                wandb.log({"train/cin_loss": cin_loss.item()}, step=step)
                wandb.log({"train/cout_loss": cout_loss.item()}, step=step)
                wandb.log({"train/bce_loss": bce_loss.item()}, step=step)
                wandb.log({"train/brightness_loss": brightness_loss.item()}, step=step)
                wandb.log({"train/loss": loss.item()}, step=step)
            
                if step > 0 and config["val_every_n_steps"] > 0 and step % config["val_every_n_steps"] == 0:
                    with torch.no_grad():
                        my_log_aggregation_network(aggregation_network, config)
                        #save_model(config, aggregation_network, optimizer, step,diffusion_extractor.unet )
                        validate(config, diffusion_extractor, aggregation_network, val_dataloader)
                step += 1

def main(args):
    config, diffusion_extractor, aggregation_network = load_models(args.config_path)
    global augs_path
    augs_path = args.save_augs_folder
    if augs_path:
        create_folder_structure(augs_path,args.overwrite)

    wandb.init(project=config["wandb_project"])

    #### LOAD COCO CAPTIONS
    import json
    caption_file = "filename_and_captions.json"
    with open(caption_file, "rt") as f:
        captions = json.load(f)
    lines = captions["filenameandcaptions"]
    caps = {x["file_name"]: x["text"][0].strip("\n") for x in lines}
    if config.get("weights_path"):
        if config["double_bottleneck_and_mix"]:
            state_dict = torch.load(config["weights_path"],map_location="cuda")["aggregation_network"]
            try:
                aggregation_network.load_state_dict(state_dict, strict=True)
            except Exception as e:
                print("WARNING !!! Loading without STRICT")
                for key in state_dict:
                   print(key)
                aggregation_network.load_state_dict(state_dict, strict=False)
#                 aggregation_network.load_state_dict(new_state_dict, strict=False)

        else:
            state_dict = torch.load(config["weights_path"],map_location="cuda")["aggregation_network"]
            new_state_dict = {}
            for key in state_dict:
                new_key = key.replace("bottleneck_layers", "high_bottleneck_layers")
                new_state_dict[new_key] = state_dict[key]
                new_key2 = key.replace("bottleneck_layers", "low_bottleneck_layers")
                new_state_dict[new_key2] = state_dict[key]
            aggregation_network.load_state_dict(new_state_dict, strict=False)

    
    if config["double_bottleneck"]:
        parameter_groups = [
            {"params": aggregation_network.red_readout.parameters(), "lr": config["lr"]},
            {"params": aggregation_network.green_readout.parameters(), "lr": config["lr"]},    
            {"params": aggregation_network.blue_readout.parameters(), "lr": config["lr"]},        
            {"params": aggregation_network.class_readout.parameters(), "lr": config["lr"]},        
            {"params": aggregation_network.high_bottleneck_layers.parameters(), "lr": config["bottleneck_lr"]},
            {"params": aggregation_network.low_bottleneck_layers.parameters(), "lr": config["bottleneck_lr"]}    
        ]
        
    if config["double_bottleneck_and_mix"]:
        parameter_groups = [
            {"params": aggregation_network.saliency_readout.parameters(), "lr": config["lr"]},        
            {"params": aggregation_network.mixing_weights_low, "lr": config["lr"]},
            {"params": aggregation_network.mixing_weights_high, "lr": config["lr"]}, 
            
            {"params": aggregation_network.red_readout.parameters(), "lr": config["lr"]},
            {"params": aggregation_network.green_readout.parameters(), "lr": config["lr"]},    
            {"params": aggregation_network.blue_readout.parameters(), "lr": config["lr"]},        
            {"params": aggregation_network.class_readout.parameters(), "lr": config["lr"]},  
            
            {"params": aggregation_network.contrast_in_readout.parameters(), "lr": config["lr"]},  
            {"params": aggregation_network.contrast_global_readout.parameters(), "lr": config["lr"]},  
            {"params": aggregation_network.brightness_readout.parameters(), "lr": config["lr"]},  
            
            {"params": aggregation_network.high_bottleneck_layers.parameters(), "lr": config["bottleneck_lr"]},
            {"params": aggregation_network.low_bottleneck_layers.parameters(), "lr": config["bottleneck_lr"]}    
        ]        
    else:
        parameter_groups = [
            {"params": aggregation_network.mixing_weights_low, "lr": config["lr"]},
            {"params": aggregation_network.mixing_weights_high, "lr": config["lr"]},        
            {"params": aggregation_network.bottleneck_layers.parameters(), "lr": config["lr"]}
        ]

    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config["weight_decay"])
    
    def list_images_in_directory(directory):
        images = []
        for filename in os.listdir(directory):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Add more extensions if needed
                #images.append(os.path.join(directory, filename))
                images.append(filename)
        return images

    # Example usage:
    val_target_directory = "../fimplenet/saliency/salicon/images/val/"
    val_image_list = list_images_in_directory(val_target_directory)
    
    print("VAL Images", len(val_image_list))
    val_img_dir = "../fimplenet/saliency/salicon/images/val/"
    val_gt_dir = "../fimplenet/saliency/salicon/maps/val/"
    val_dataset = SaliconDataset(val_img_dir, val_gt_dir, None, val_image_list[:100])
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    #---------MS COCO------------------
    # Load COCO dataset
    #coco_dataset = CocoDetection(root='../style-transfer/train2014', annFile='../style-transfer/instances_train2014.json', transform=transform)
    # Specify paths to the COCO dataset and annotation file
    root = '../style-transfer/train2014'
    #annFile = '../style-transfer/instances_train2014.json'
    annFile = '../style-transfer/filtered_annotations.json' #ONLY SALICON IMAGES

    # Load COCO dataset
    coco_dataset = CocoDetection(root=root, annFile=annFile)
    global coco
    coco = COCO(annFile)
    # Create custom dataset
    sal_gt_dir = "../fimplenet/saliency/salicon/maps/train/"
    custom_dataset = CustomCocoDataset(coco_dataset,sal_gt_dir, transform=transform)
    # Create DataLoader
    train_dataloader = DataLoader(custom_dataset, batch_size=config["batch_size"], shuffle=True)

    train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader,caps)

if __name__ == "__main__":
    # python3 train_generic.py --config_path configs/train.yaml
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/train.yaml")
    parser.add_argument("--save_augs_folder", type=str, help="Path to save the augmentaions", default="")
    parser.add_argument("--overwrite", type=int, help="Path to save the augmentaions", default=0)

    args = parser.parse_args()
    main(args)
