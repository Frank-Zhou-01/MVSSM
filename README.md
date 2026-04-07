# MVSSM
## MVSSM: Motion-aware Visual State Space Model for Efficient Video Deblurring

<hr />

> **Abstract**: *Video deblurring remains a challenging task due to the complex and spatially varying motion blur present in real-world scenes.  Recent state space models (SSMs) have shown great promise in modeling long-range dependencies with linear complexity;  however, their fixed scanning strategies often fail to capture motion-aware contextual information effectively. To address this limitation, we propose MVSSM, a Motion-aware Visual State Space Model that dynamically adapts its feature scanning direction based on estimated motion trajectories. Specifically, our Motion-aware Scanning Block (MSB) leverages optical flow to determine the dominant motion direction and selects an optimal scanning path—horizontal, vertical, or diagonal—to align feature propagation with actual motion patterns. Furthermore, we introduce a Multi-scale Dynamic Feature Fusion (MDFF) module that enhances spatiotemporal representation through pixel-wise dynamic convolutions across multiple scales. Experimental results on multiple synthetic and real-world datasets demonstrate that the proposed method not only substantially reduces computational burden but also achieves state-of-the-art performance in video deblurring.*
<hr />

This repository is the official PyTorch implementation of our CVPR2026 findings paper "MVSSM: Motion-aware Visual State Space Model for Efficient Video Deblurring".
