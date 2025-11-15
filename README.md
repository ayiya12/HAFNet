# HAFNet: Hybrid Attention Fusion Network for Remote Sensing Pansharpening

Abstract: Deep learning–based pansharpening methods for remote sensing have advanced rapidly in recent years. However, current methods still face three limitations that directly affect reconstruction quality.  Content adaptivity is often implemented as an isolated step, which prevents effective interaction across scales and feature domains. Dynamic multi-scale mechanisms also remain constrained, since their scale selection is usually guided by global statistics and ignores regional heterogeneity. Moreover, frequency and spatial cues are commonly fused in a static manner, leading to an imbalance between global structural enhancement and local texture preservation. To address these issues, we design three complementary modules. We utilize the Adaptive Convolution Unit (ACU) to generate content-aware kernels through local feature clustering, thereby achieving fine-grained adaptation to diverse ground structures. We also develop the Multi-Scale Receptive Field Selection Unit (MSRFU), a module providing flexible scale modeling by selecting informative branches at varying receptive fields. Meanwhile, we incorporate the Frequency–Spatial Attention Unit (FSAU), designed to dynamically fuse spatial representations with frequency information. This effectively strengthens detail reconstruction while minimizing spectral distortion. Specifically, we propose the Hybrid Attention Fusion Network (HAFNet), which employs the Hybrid Attention-Driven Residual Block (HARB) as the fundamental utility to dynamically integrate the above three specialized components. This design enables dynamic content adaptivity, multi-scale responsiveness, and cross-domain feature fusion within a unified framework. Experiments on public benchmarks confirm the effectiveness of each component and demonstrate HAFNet's state-of-the-art performance.

## Getting Started

### Dataset

- Datasets are used from pansharpening： [liangjiandeng/PanCollection](https://github.com/liangjiandeng/PanCollection).

**Please prepare a Docker environment with CUDA support:**

- Ensure you have Docker installed on your system.
- To enable CUDA support within the Docker environment, refer to the official Docker documentation for setting up GPU acceleration: Docker GPU setup: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

If you cannot use Docker, you can also set up the environment manually. However, you may run into issues with the dependencies.

1. **Clone the repo and its submodules:**
   
   ```bash
   git clone --recurse-submodules https://github.com/duanyll/HAFNet.git
   ```

2. **Edit mount point for datasets in `.devcontainer/devcontainer.json`:**
    - Locate the `.devcontainer/devcontainer.json` file within the cloned repo.
    - Specify the path to your datasets on your host machine by adjusting the `mounts` configuration in the file.

3. **Reopen the repo in VS Code devcontainer:**
    - Open the cloned repo in VS Code.
    - When prompted, select "Reopen in Container" to activate the devcontainer environment.
    - It may take serval minutes when pulling the base PyTorch image and install requirements for the first time.

4. **Install pacakges and build native libraries**
   - If you are using the devcontainer, you can skip this step, vscode will automatically run the script.
   
   ```bash
   bash ./build.sh
   ```

5. **Train the model:**
   
   ```bash
   python -m hafnet.scripts.train cannet wv3
   ```

   - Replace `cannet` with other networks available in the `hafnet/models` directory.
   - Replace `wv3` with other datasets defined in `presets.json`.
   - Results are placed in the `runs` folder.

## Additional Information

**Pretrained weights:**

- Pre-trained weights can be found in the `weights` folder.

**Datasets:**
- Datasets are used from the repo [liangjiandeng/PanCollection](https://github.com/liangjiandeng/PanCollection).

**Metrics:**
- Metrics are obtained using tools from [liangjiandeng/DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox) (specifically, the `02-Test-toolbox-for-traditional-and-DL(Matlab)` directory).
