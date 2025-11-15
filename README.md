# HAFNet: Hybrid Attention Fusion Network for Remote Sensing Pansharpening

![framework](images/framework.png)

Abstract: Deep learning–based pansharpening methods for remote sensing have advanced rapidly in recent years. However, current methods still face three limitations that directly affect reconstruction quality.  Content adaptivity is often implemented as an isolated step, which prevents effective interaction across scales and feature domains. Dynamic multi-scale mechanisms also remain constrained, since their scale selection is usually guided by global statistics and ignores regional heterogeneity. Moreover, frequency and spatial cues are commonly fused in a static manner, leading to an imbalance between global structural enhancement and local texture preservation. To address these issues, we design three complementary modules. We utilize the Adaptive Convolution Unit (ACU) to generate content-aware kernels through local feature clustering, thereby achieving fine-grained adaptation to diverse ground structures. We also develop the Multi-Scale Receptive Field Selection Unit (MSRFU), a module providing flexible scale modeling by selecting informative branches at varying receptive fields. Meanwhile, we incorporate the Frequency–Spatial Attention Unit (FSAU), designed to dynamically fuse spatial representations with frequency information. This effectively strengthens detail reconstruction while minimizing spectral distortion. Specifically, we propose the Hybrid Attention Fusion Network (HAFNet), which employs the Hybrid Attention-Driven Residual Block (HARB) as the fundamental utility to dynamically integrate the above three specialized components. This design enables dynamic content adaptivity, multi-scale responsiveness, and cross-domain feature fusion within a unified framework. Experiments on public benchmarks confirm the effectiveness of each component and demonstrate HAFNet's state-of-the-art performance.

## Experiment results

- Quantitative results

  ![reslut](images/result.png)

- Visual results

  ![visual](images/visual.png)

## Getting Started

### **Environment** 

- PyTorch 2.1.0

- Python 3.10

- CUDA 12.1

### Installation

```
# Clone the repository
git clone --recurse-submodules https://github.com/ayiya12/HAFNet.git
cd HAFNet

# Install Python dependencies
pip install -r requirements.txt

# Build CUDA/C++ extensions
bash build.sh
```

### Dataset

- Download datasets from: [liangjiandeng/PanCollection](https://github.com/liangjiandeng/PanCollection).
- Edit `presets.json` to configure dataset paths

### Training

```
# Train on WV3 dataset
python -m hafnet.scripts.train hafnet wv3  
  
# Train on GF2 dataset
python -m hafnet.scripts.train hafnet gf2  
  
# Train on QB dataset
python -m hafnet.scripts.train hafnet qb
```

Training results are saved in `runs/<exp_name>/` directory.

### Testing

Configure test dataset paths in `presets.json` for `test_reduced_data` and `test_origscale_data`.

```
# Test on wv3 reduced resolution dataset
python -m hafnet.scripts.test hafnet runs/hafnet_wv3/weights/final.pth dataset/wv3/test_wv3_multiExm1.h5 --preset wv3 --output_dir results  
  
# Test on wv3 original scale dataset
python -m hafnet.scripts.test hafnet runs/hafnet_wv3/weights/final.pth dataset/wv3/test_wv3_OrigScale_multiExm1.h5 --preset wv3 --output_dir results
```

Test results are saved as MAT files.

## Additional Information

**Pretrained weights:**

- Pretrained model weights for WV3 dataset are available in the `weights/` folder. 
- The checkpoint represents a model trained for 500 iterations on the WV3 dataset.  

**Sample Results:**

- Sample test results on WV3 reduced resolution dataset are included in the repository. 
- These results demonstrate the model's performance on the WV3 test set and can be used as reference outputs for validation. 

**Evaluation:**

- Metrics can be computed using tools from [DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox). 
