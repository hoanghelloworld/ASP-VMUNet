# ASP-VMUNet: Atrous Shifted Parallel Vision Mamba U-Net for Skin Lesion Segmentation

### Authors:
- **[Muyi Bao](https://github.com/BaoBao0926/BaoBao0926.github.io)**, **Shuchang Lyu**, **Zhaoyang Xu**, **Qi Zhao**, **Changyu Zeng**, **Wenpei Bai**, **Guangliang Cheng**

### NEWS:
- Code is coming soon
- 2024.12.16: The repository is created.


# Abstract
Skin lesion segmentation is a critical challenge in computer vision, essential to accurately separate pathological features from healthy skin for diagnostics. Traditional Convolutional Neural Networks (CNNs) are limited by narrow receptive fields, and Transformers face significant computational burdens. This paper presents a novel skin lesion segmentation framework, the Atrous Shifted Parallel Vision Mamba UNet (ASP-VMUNet), which integrates the efficient and scalable Mamba architecture to overcome limitations in traditional CNNs and computationally demanding Transformers. The framework introduces an atrous scan technique that minimizes background interference and expands the receptive field, enhancing Mamba's scanning capabilities. Additionally, the inclusion of a Parallel Vision Mamba (PVM) layer and a shift round operation optimizes feature segmentation and fosters rich inter-segment information exchange. A supplementary CNN branch with an Selective-Kernel (SK) Block further refines the segmentation by blending local and global contextual information. Tested on four benchmark datasets (ISIC16/17/18 and PH2), ASP-VMUNet demonstrates superior performance in skin lesion segmentation, validated by comprehensive ablation studies. This approach not only advances medical image segmentation but also highlights the benefits of hybrid architectures in medical imaging technology.


<img src="https://github.com/BaoBao0926/ASP-VMUNet/blob/main/figure/ASPVMUnet.png" alt="ASP-VMUNet" width="700"/> 
<img src="https://github.com/BaoBao0926/ASP-VMUNet/blob/main/figure/ASP_Layer.png" alt="ASP_Layer" width="700"/>

<img src="https://github.com/BaoBao0926/ASP-VMUNet/blob/main/figure/SE.png" alt="SE" width="300"/>  <img src="https://github.com/BaoBao0926/ASP-VMUNet/blob/main/figure/SK.png" alt="SK" width="350"/>  <img src="https://github.com/BaoBao0926/ASP-VMUNet/blob/main/figure/AMB.png" alt="ASB" width="340"/>



# Experimental Results
### Comparison on Benchmark
<img src="https://github.com/BaoBao0926/ASP-VMUNet/blob/main/figure/ComparsionResult.png" alt="ASP_Layer" width="600"/>

### Ablation Study of Component
<img src="https://github.com/BaoBao0926/ASP-VMUNet/blob/main/figure/AS_component.png" alt="AS_component" width="600"/>

### Ablation Study of Component
<img src="https://github.com/BaoBao0926/ASP-VMUNet/blob/main/figure/AS_atrousstep.png" alt="AS_atrousstep" width="600"/>

### Ablation Study of Atrous Scan and The Number of Encoder Layers
<img src="https://github.com/BaoBao0926/ASP-VMUNet/blob/main/figure/AS_atrousscan_layer.png" alt="AS_atrousscan_layer" width="600"/>



















