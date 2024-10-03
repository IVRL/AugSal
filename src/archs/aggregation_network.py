import numpy as np
import torch
from torch import nn
from archs.detectron2.resnet import ResNet, BottleneckBlock

class AggregationNetwork(nn.Module):
    """
    Module for aggregating feature maps across time and space.
    Design inspired by the Feature Extractor from ODISE (Xu et. al., CVPR 2023).
    https://github.com/NVlabs/ODISE/blob/5836c0adfcd8d7fd1f8016ff5604d4a31dd3b145/odise/modeling/backbone/feature_extractor.py
    """
    def __init__(
            self, 
            feature_dims, 
            device, 
            projection_dim=384, 
            num_norm_groups=32,
            num_res_blocks=1, 
            save_timestep=[],
            num_timesteps=None,
            timestep_weight_sharing=False,
            double_bottleneck = False,
        ):
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims = feature_dims    
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        self.device = device
        self.save_timestep = save_timestep
        self.double_bottleneck= double_bottleneck
        print("double_bottleneck",double_bottleneck)
        if double_bottleneck:
                self.low_bottleneck_layers = nn.ModuleList()
                self.high_bottleneck_layers = nn.ModuleList()
                
        self.mixing_weights_names = []
        for l, feature_dim in enumerate(self.feature_dims):
#             bottleneck_layer = nn.Sequential(
#                 *ResNet.make_stage(
#                     BottleneckBlock,
#                     num_blocks=num_res_blocks,
#                     in_channels=feature_dim,
#                     bottleneck_channels=projection_dim // 4,
#                     out_channels=projection_dim,
#                     norm="GN",
#                     num_norm_groups=num_norm_groups
#                 )
#             )
            if double_bottleneck:
                bottleneck_layer = nn.Sequential(
                    *ResNet.make_stage(
                        BottleneckBlock,
                        num_blocks=num_res_blocks,
                        in_channels=feature_dim,
                        bottleneck_channels=projection_dim // 4,
                        out_channels=projection_dim,
                        norm="GN",
                        num_norm_groups=num_norm_groups
                    )
                )            
                self.low_bottleneck_layers.append(bottleneck_layer)
                bottleneck_layer_high = nn.Sequential(
                    *ResNet.make_stage(
                        BottleneckBlock,
                        num_blocks=num_res_blocks,
                        in_channels=feature_dim,
                        bottleneck_channels=projection_dim // 4,
                        out_channels=projection_dim,
                        norm="GN",
                        num_norm_groups=num_norm_groups
                    )
                )            
                self.high_bottleneck_layers.append(bottleneck_layer_high)               
            else:
                bottleneck_layer = nn.Sequential(
                    *ResNet.make_stage(
                        BottleneckBlock,
                        num_blocks=num_res_blocks,
                        in_channels=feature_dim,
                        bottleneck_channels=projection_dim // 4,
                        out_channels=projection_dim,
                        norm="GN",
                        num_norm_groups=num_norm_groups
                    )
                )            
                self.bottleneck_layers.append(bottleneck_layer)
            for t in save_timestep:
                # 1-index the layer name following prior work
                self.mixing_weights_names.append(f"timestep-{save_timestep}_layer-{l+1}")
                    
        if double_bottleneck:
            self.high_bottleneck_layers = self.high_bottleneck_layers.to(device)
            self.low_bottleneck_layers = self.low_bottleneck_layers.to(device)
        else:
            self.bottleneck_layers = self.bottleneck_layers.to(device)
            print("timesteps ",len(save_timestep))
            print("bottleneck_layers ",len(self.bottleneck_layers))
            print("bottleneck_layers0 ",self.bottleneck_layers[0])
            print("bottleneck_layers1 ",self.bottleneck_layers[1])
        if double_bottleneck:# TODO: CHANGE TO config["double_bottleneck_and_mix"]
                    #mixing_weights_low = torch.ones(len(self.bottleneck_layers) * len(save_timestep))
            mixing_weights_low = torch.ones(len(self.high_bottleneck_layers) * len(save_timestep))
            self.mixing_weights_low = nn.Parameter(mixing_weights_low.to(device))
            mixing_weights_high = torch.ones(len(self.low_bottleneck_layers) * len(save_timestep))
            self.mixing_weights_high = nn.Parameter(mixing_weights_high.to(device))            
        else:
            #mixing_weights_low = torch.ones(len(self.bottleneck_layers) * len(save_timestep))
            mixing_weights_low = torch.tensor([1., 1., 1., 0.5, 0.3, 0.1, 0.1, 0.3, 0.5, 1., 1., 1.])
            self.mixing_weights_low = mixing_weights_low.to(device)
            #mixing_weights_high= torch.ones(len(self.bottleneck_layers) * len(save_timestep))
            mixing_weights_high = torch.tensor([ 0.1, 0.3, 0.5,1., 1., 1.,1., 1., 1., 0.5, 0.3, 0.1 ])
            self.mixing_weights_high = mixing_weights_high.to(device)      
        print("mixing_weights_low", self.mixing_weights_low)
        print("mixing_weights_high", self.mixing_weights_high)

        self.red_readout = MLPReadout(384,1).to(device)
        self.green_readout = MLPReadout(384,1).to(device)
        self.blue_readout = MLPReadout(384,1).to(device)
        if True:
            self.contrast_in_readout = MLPReadout(384,1).to(device)
            self.contrast_global_readout = MLPReadout(387,1).to(device)
            self.brightness_readout = MLPReadout(384,1).to(device)
        
        self.class_readout = MLPReadout(384,91).to(device)
        self.saliency_readout = ConvReadout().to(device)
#         mixing_weights_low = torch.tensor([1.])
#         self.a_contrast = nn.Parameter(mixing_weights_high.to(device))  
#         self.a_brightness = nn.Parameter(mixing_weights_high.to(device))  
#         self.a_r = nn.Parameter(mixing_weights_high.to(device))  
#         self.a_g = nn.Parameter(mixing_weights_high.to(device))  
#         self.a_b = nn.Parameter(mixing_weights_high.to(device))  
        
        
        
    def forward(self, batch,timestep_embedding):
        """
        Assumes batch is shape (B, C, H, W) where C is the concatentation of all layer features.
        """
        low_output_feature = None
        start = 0
        mixing_weights_low = torch.nn.functional.softmax(self.mixing_weights_low)
        for i in range(len(mixing_weights_low)):
            if self.double_bottleneck:
                bottleneck_layer = self.low_bottleneck_layers[i % len(self.feature_dims)]
            else:
            # Share bottleneck layers across timesteps
                bottleneck_layer = self.bottleneck_layers[i % len(self.feature_dims)]
            # Chunk the batch according the layer
            # Account for looping if there are multiple timesteps
            end = start + self.feature_dims[i % len(self.feature_dims)]
            feats = batch[:, start:end, :, :]
            start = end
            # Downsample the number of channels and weight the layer
#             print("NOTTLE",bottleneck_layer)
            bottlenecked_feature = bottleneck_layer[0](feats,timestep_embedding)
            bottlenecked_feature = mixing_weights_low[i] * bottlenecked_feature
            if low_output_feature is None:
                low_output_feature = bottlenecked_feature
            else:
                low_output_feature += bottlenecked_feature
        high_output_feature = None
        start = 0
        mixing_weights_high = torch.nn.functional.softmax(self.mixing_weights_high)
        for i in range(len(mixing_weights_high)):
            if self.double_bottleneck:
                bottleneck_layer = self.high_bottleneck_layers[i % len(self.feature_dims)]
            else:            
                # Share bottleneck layers across timesteps
                bottleneck_layer = self.bottleneck_layers[i % len(self.feature_dims)]
            # Chunk the batch according the layer
            # Account for looping if there are multiple timesteps
            end = start + self.feature_dims[i % len(self.feature_dims)]
            feats = batch[:, start:end, :, :]
            start = end
            # Downsample the number of channels and weight the layer
#             print("NOTTLE",bottleneck_layer)
            bottlenecked_feature = bottleneck_layer[0](feats,timestep_embedding)
            bottlenecked_feature = mixing_weights_high[i] * bottlenecked_feature
            if high_output_feature is None:
                high_output_feature = bottlenecked_feature
            else:
                high_output_feature += bottlenecked_feature                
                
        return high_output_feature,low_output_feature
    


class FeatureMapStandardizer(nn.Module):
    def __init__(self, in_channels):
        super(FeatureMapStandardizer, self).__init__()
        
        # 1x1 Convolution layer with 96 filters
        self.conv1x1 = nn.Conv2d(in_channels, 96, kernel_size=1)
        
        # Linear layer for timestep embedding
        self.linear = nn.Linear(1280, 96)
        
        # 3x3 Convolution layer with 96 filters
        self.conv3x3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        
        # 1x1 Convolution layer with 384 filters
        self.conv1x1_out = nn.Conv2d(96, 384, kernel_size=1)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x,timestep_embedding):
        # 1x1 Convolution
        out_conv1x1 = self.conv1x1(x)
        
        # Linear layer and ReLU activation for timestep embedding
        te = self.relu(self.linear(timestep_embedding))
        
        # Element-wise addition
        out_add = out_conv1x1 + te#.view(1, 96, 1, 1)
        
        # ReLU activation
        out_add_relu = self.relu(out_add)
        
        # 3x3 Convolution
        out_conv3x3 = self.conv3x3(out_add_relu)
        
        # ReLU activation
        out_conv3x3_relu = self.relu(out_conv3x3)
        
        # 1x1 Convolution
        out_conv1x1_out = self.conv1x1_out(out_conv3x3_relu)
        
        # Element-wise addition with original input
        out_final = x + out_conv1x1_out
        
        # ReLU activation
        out_final_relu = self.relu(out_final)
        
        return out_final_relu
    
class MLPReadout(nn.Module):
    def __init__(self,num_in=384, num_out=1):
        super(MLPReadout, self).__init__()
        
        # 3x3 convolution with 128 output channels
        self.conv1 = nn.Conv2d(num_in, 128, kernel_size=3, padding=1)
        self.silu1 = nn.SiLU()  # SiLU (Sigmoid Linear Unit) activation function
        
        # 3x3 convolution with 32 output channels
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.silu2 = nn.SiLU()  # SiLU (Sigmoid Linear Unit) activation function
        
        # MLP with 32 input features and 3 output features
        self.mlp = nn.Sequential(
            nn.Linear(32 * 64 * 64, 128),  # Assuming input image size is 64x64
            nn.SiLU(),  # SiLU (Sigmoid Linear Unit) activation function
            nn.Linear(128, num_out)
        )

    def forward(self, x):
        # Apply conv1 and silu1
        x = self.conv1(x)
        x = self.silu1(x)
        
        # Apply conv2 and silu2
        x = self.conv2(x)
        x = self.silu2(x)
        
        # Flatten the output before passing it to MLP
        x = x.view(x.size(0), -1)
        
        # Apply MLP
        x = self.mlp(x)
        
        return x
    
class ConvReadout(nn.Module):
    def __init__(self, in_channels=384+384):
        super(ConvReadout, self).__init__()
        
        
        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        
        # Linear layer for timestep embedding
        self.linear = nn.Linear(1280, 96)
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)       
        self.tanh = nn.Tanh()       
        self.sigmoid = nn.Sigmoid()
        self.conv_high = nn.Conv2d(int(in_channels/2), int(in_channels/2), kernel_size=1)
        self.conv_low = nn.Conv2d(int(in_channels/2), int(in_channels/2), kernel_size=1)
        
        # 3x3 convolution with 128 output channels
        self.conv1 = nn.Conv2d(in_channels, 384, kernel_size=3, padding=1)
        self.silu1 = nn.SiLU()  # SiLU (Sigmoid Linear Unit) activation function
        # 3x3 convolution with 32 output channels
        self.conv2 = nn.Conv2d(384, 192, kernel_size=3, padding=1)
                
        self.conv_high_skip = nn.Conv2d(192+384, 192, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.conv_low_skip = nn.Conv2d(64+384, 192, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 8, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(8, 1, kernel_size=3, padding=1)

 

    def forward(self, highf, lowf):
        highf = self.tanh(self.conv_high(highf))
        lowf = self.tanh(self.conv_low(lowf))
        x = torch.cat([highf, lowf], dim=1)
        # Apply conv1 and silu1
        x = self.conv1(x)
        x = self.silu1(x)
        
        # Apply conv2 an
        x = self.conv2(x)
        x = self.silu1(x)
        x = torch.cat([highf, x], dim=1)
        x = self.conv_high_skip(x)
        x = self.silu1(x)
        
        x = self.conv3(x)
        x = self.silu1(x)
        x = torch.cat([lowf, x], dim=1)
        x = self.conv_low_skip(x)
        x = self.silu1(x)        

        x = self.conv4(x)
        x = self.silu1(x)         
        x = self.conv5(x)
        x = self.silu1(x)
        x = self.conv6(x)
        x = self.sigmoid(x)        
        
        return x