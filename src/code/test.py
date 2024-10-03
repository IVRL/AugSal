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
from archs.correspondence_utils import process_image
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
import train_generic
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np

def gaussian_kernel(kernel_size, sigma):
    """
    Create a 2D Gaussian kernel.
    """
    # Create a grid of points centered around the middle
    grid = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(grid, grid)
    
    # Calculate the 2D Gaussian kernel
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    
    # Normalize the kernel
    kernel /= kernel.sum()
    
    return kernel

def gaussian_blur(input_tensor, kernel_size=21, sigma=7.0):
    """
    Apply Gaussian blur to the input tensor.
    """
    # Convert the kernel to a PyTorch tensor
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda()
    
    # Apply the Gaussian blur using convolution
    blurred_tensor = F.conv2d(input_tensor, kernel, padding=kernel_size//2)
    
    return blurred_tensor

# Set up transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
transform64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
def test(config, diffusion_extractor, aggregation_network, val_dataloader):
    tic = time.time()
    device = config.get("device", "cuda")

    i = 0
    with torch.no_grad():
        for (img,img_id) in val_dataloader:
            img = img.to(device)
            img_id =img_id[0]

            diffusion_hyperfeats_high,diffusion_hyperfeats_low = train_generic.get_hyperfeats(diffusion_extractor, aggregation_network, img)
            print("diffusion_hyperfeats_high",diffusion_hyperfeats_high.shape)
            print("diffusion_hyperfeats_low",diffusion_hyperfeats_low.shape)

            blur_map = aggregation_network.saliency_readout(diffusion_hyperfeats_high,diffusion_hyperfeats_low)
            pred_map = torch.nn.functional.interpolate(blur_map, size=(480,640), mode="bilinear")

            # Blurring
            blur_map = pred_map.mean(dim=0).unsqueeze(0)

            blur_map = blur_map /blur_map.max()
            # Rescale the tensor to the range [0, 1] if it's not already
            tensor_image = torch.clamp(blur_map, 0, 1) 
            # Convert tensor to PIL image
            pil_image = torchvision.transforms.ToPILImage()(tensor_image.squeeze(0))

            # Save the PIL image
            pil_image.save("./outputs/91modelblurred/"+img_id.split(".jpg")[0]+".png","PNG")   
            if i%100 == 0:
                print("Completed images: ", i)
            print(i)
            i+=1
    return 0




def main(args):
    config, diffusion_extractor, aggregation_network = load_models(args.config_path)



    if config.get("weights_path"):
        if config["double_bottleneck_and_mix"]:
            state_dict = torch.load(config["weights_path"],map_location="cuda")["aggregation_network"]
            try:
                aggregation_network.load_state_dict(state_dict, strict=True)
            except Exception as e:
                print("e")
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

    

    def list_images_in_directory(directory):
        images = []
        for filename in os.listdir(directory):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Add more extensions if needed
                #images.append(os.path.join(directory, filename))
                images.append(filename)
        return images

    # Example usage:
    test_target_directory = "../saliency/dataset/images/test/"
    test_image_list = list_images_in_directory(test_target_directory)
    
    print("test Images", len(test_image_list))
    test_img_dir = "../saliency/dataset/images/test/"
    test_dataset = SaliconTest(test_img_dir, test_image_list)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    test(config, diffusion_extractor, aggregation_network, test_dataloader)

if __name__ == "__main__":
    # python3 train_generic.py --config_path configs/train.yaml
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/test.yaml")
    args = parser.parse_args()
    main(args)