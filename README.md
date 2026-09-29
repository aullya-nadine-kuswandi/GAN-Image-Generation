# Conditional GAN for Overhead Image Generation

Class-conditional image generation with GANs on the OverheadMNIST dataset (**car** vs **plane**, 28×28 grayscale). The project starts from a simple MLP-based conditional GAN, then improves it with a convolutional architecture (cDCGAN), label smoothing, and TTUR. Each step is evaluated with Fréchet Inception Distance (FID).

**Result:** FID reduced from **215.79 to 79.58** (about 63% improvement).

## Key Objectives

- **Data Preparation:** Load the `car` and `plane` classes from OverheadMNIST, convert to 28×28 grayscale, and inspect class balance (7,116 vs 7,124 training images).
- **Preprocessing & Scaling:** Normalize pixel values to [-1, 1] to match the Generator's final `tanh` activation.
- **Baseline cGAN:** Implement a fully-connected (MLP) Generator and Discriminator conditioned on class labels via embeddings.
- **Improved cDCGAN:**
  - Replace dense layers with `Conv2DTranspose` (upsampling) and strided `Conv2D` (feature extraction) to preserve spatial structure.
  - Apply label smoothing (real target = 0.9) to prevent an overconfident Discriminator.
  - Tune learning rates with the Two Time-Scale Update Rule (TTUR).
- **Quantitative Evaluation:** Measure generation quality with Fréchet Inception Distance (FID) using InceptionV3 features.

## Dataset

[OverheadMNIST](https://www.kaggle.com/datasets/datamunge/overheadmnist) on Kaggle, using only the `car` and `plane` classes.

| Split | Car | Plane | Total |
|-------|-----|-------|-------|
| Train | 7,116 | 7,124 | 14,240 |
| Test  | 896 | 888 | 1,784 |

The baseline flattens images to 784-dim vectors, while the cDCGAN keeps the `(28, 28, 1)` shape.

## Models

### 1. Baseline: MLP Conditional GAN
- **Generator:** noise (100-d) + label embedding → Dense 128 → 256 → 512 → 1024 → 784 (`tanh`), LeakyReLU(0.2)
- **Discriminator:** flattened image + label embedding → Dense 512 → 1024 → 1024 → 512 → 1 (logit), LeakyReLU(0.2)

### 2. Modified: Conditional DCGAN + Label Smoothing
- **Generator:** noise + label embedding → Dense (7×7×128) → Conv2DTranspose 128 → Conv2DTranspose 64 (each with BatchNorm + LeakyReLU) → Conv2D 1 (`tanh`)
- **Discriminator:** image + label embedding (as an extra channel) → Conv2D 64 → Conv2D 128 (LeakyReLU + Dropout 0.3) → Flatten → Dense 1 (logit)
- **Label smoothing:** real target 1.0 → 0.9

### 3. Tuned: cDCGAN + TTUR
Same architecture as above, with separate learning rates for Generator and Discriminator (Two Time-Scale Update Rule).

## Training Setup

- Loss: Binary Cross-Entropy (`from_logits=True`)
- Optimizer: Adam (β₁ = 0.5)
- Batch size 64, 100 epochs, latent dim 100
- TTUR configs tested (G lr, D lr): (1e-4, 4e-4), (2e-4, 2e-4), (5e-5, 4e-4)

## Results

| Model | FID ↓ |
|-------|-------|
| Baseline (MLP cGAN) | 215.79 |
| Modified (cDCGAN + label smoothing) | 84.39 |
| **Modified + TTUR (G 1e-4, D 4e-4)** | **79.58** |

**TTUR sweep**

| G lr | D lr | FID |
|------|------|-----|
| 1e-4 | 4e-4 | **79.58** |
| 2e-4 | 2e-4 | 84.42 |
| 5e-5 | 4e-4 | 96.45 |

## Key Findings

- **Baseline:** the Discriminator dominated (D-loss kept dropping while G-loss kept rising). Flattening images to 784 pixels discards spatial structure, so the outputs were blurry.
- **cDCGAN + label smoothing:** convolutional layers preserve spatial patterns, and label smoothing keeps the Discriminator from becoming overconfident. FID dropped by about 61%, and the losses became balanced and stable.
- **TTUR:** a slightly faster Discriminator (D lr > G lr) gave the best FID. Too large a gap (5e-5 vs 4e-4) hurt the Generator because it couldn't keep up.

## Conclusion

This project shows that GAN quality depends heavily on architecture and training balance, not just on training longer.

- The MLP baseline treats each pixel independently, which loses spatial structure. The Discriminator quickly overpowered the Generator, and the generated images stayed blurry (FID 215.79).
- Switching to a convolutional cDCGAN with label smoothing gave the biggest improvement, cutting FID to 84.39 and making training much more balanced.
- TTUR gave a smaller but consistent gain (FID 79.58). Giving the Discriminator a somewhat higher learning rate than the Generator worked best, while a gap that was too large made things worse.
- Overall, combining a convolutional architecture, label smoothing, and TTUR reduced FID by about 63% compared to the baseline and produced the most stable training.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Metrics & Math:** SciPy (`sqrtm`), InceptionV3 (Keras Applications)
- **Data Handling & Visualization:** Pandas, NumPy, Matplotlib, Pillow
- **Environment:** Kaggle Notebooks (Tesla T4 GPU)
