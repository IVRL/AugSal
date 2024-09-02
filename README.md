# Data Augmentation for Saliency Prediction via Latent Diffusion (ECCV 2024)
This is the GitHub repository for Data Augmentation for Saliency Prediction via Latent Diffusion paper in ECCV 2024, Milano, Italy

![teaser-5](https://github.com/user-attachments/assets/f0228a97-f1f8-4d09-a687-b4887abe7ee3)


Saliency prediction models are constrained by the limited diversity and quantity of labeled data. Standard data augmentation techniques such as rotating and cropping alter scene composition, affecting saliency. We propose a novel data augmentation method for deep saliency prediction that edits natural images while preserving the complexity and variability of real-world scenes. Since saliency depends on high-level and low-level features, our approach involves learning both by incorporating photometric and semantic attributes such as color, contrast, brightness, and class. To that end, we introduce a saliency-guided cross-attention mechanism that enables targeted edits on the photometric properties, thereby enhancing saliency within specific image regions. Experimental results show that our data augmentation method consistently improves the performance of various saliency models. Moreover, leveraging the augmentation features for saliency prediction yields superior performance on publicly available saliency benchmarks. Our predictions align closely with human visual attention patterns in the edited images, as validated by a user study. 

Please visit our visualizations at https://augsal.github.io/.
