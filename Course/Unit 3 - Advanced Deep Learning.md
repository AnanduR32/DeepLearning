# Unit III — Advanced Deep Learning

> **Course Code:** 25CSA543A — Deep Learning for AI  
> **Target Audience:** College Freshmen / Beginners in AI  
> **Core Objective:** Master generative AI (RBMs, Autoencoders, VAEs, GANs), Reinforcement Learning (DQN), and advanced training & deployment techniques in PyTorch and TensorFlow.

---

## Table of Contents
1. [Generative vs. Discriminative Models](#1-generative-vs-discriminative-models)
   - [Analogy: The Art Critic vs. The Painter](#analogy-the-art-critic-vs-the-painter)
   - [Joint Probability vs. Conditional Probability](#joint-probability-vs-conditional-probability)
   - [Energy-Based Models](#energy-based-models)
2. [Restricted Boltzmann Machines (RBM)](#2-restricted-boltzmann-machines-rbm)
   - [Architecture: Visible and Hidden Layers](#architecture-visible-and-hidden-layers)
   - [Why "Restricted"? (Conditional Independence)](#why-restricted-conditional-independence)
   - [Contrastive Divergence (CD-k) Training](#contrastive-divergence-cd-k-training)
3. [Deep Belief Networks (DBN)](#3-deep-belief-networks-dbn)
   - [Greedy Layer-Wise Pretraining](#greedy-layer-wise-pretraining)
   - [DBN vs. Single RBM](#dbn-vs-single-rbm)
4. [Autoencoders](#4-autoencoders)
   - [Architecture: Encoder, Bottleneck, Decoder](#architecture-encoder-bottleneck-decoder)
   - [Types: Undercomplete, Sparse, Denoising, Contractive](#types-undercomplete-sparse-denoising-contractive)
   - [Autoencoders vs. PCA (Nonlinear Manifolds)](#autoencoders-vs-pca-nonlinear-manifolds)
5. [Variational Autoencoders (VAE)](#5-variational-autoencoders-vae)
   - [Why Regular Autoencoders Fail at Generation](#why-regular-autoencoders-fail-at-generation)
   - [Probabilistic Latent Space ($\mu$ and $\sigma$)](#probabilistic-latent-space-\mu-and-\sigma)
   - [The Reparameterization Trick](#the-reparameterization-trick)
   - [VAE Loss Function (Reconstruction + KL Divergence)](#vae-loss-function-reconstruction--kl-divergence)
6. [Reinforcement Learning & Deep Q-Networks (DQN)](#6-reinforcement-learning--deep-q-networks-dqn)
   - [Core Concepts: Agent, Environment, State, Action, Reward, Policy](#core-concepts-agent-environment-state-action-reward-policy)
   - [The Exploration vs. Exploitation Dilemma](#the-exploration-vs-exploitation-dilemma)
   - [Q-Learning & The Bellman Equation](#q-learning--the-bellman-equation)
   - [DQN Innovations: Experience Replay & Target Networks](#dqn-innovations-experience-replay--target-networks)
7. [Generative Adversarial Networks (GANs)](#7-generative-adversarial-networks-gans)
   - [The Counterfeiter vs. Police Game](#the-counterfeiter-vs-police-game)
   - [Minimax Objective Function](#minimax-objective-function)
   - [Architectures: DCGAN and Conditional GAN (cGAN)](#architectures-dcgan-and-conditional-gan-cgan)
   - [Mode Collapse & Stability Tricks](#mode-collapse--stability-tricks)
8. [Training Challenges & Solutions](#8-training-challenges--solutions)
   - [Vanishing & Exploding Gradients](#vanishing--exploding-gradients)
   - [Batch Normalization Deep-Dive](#batch-normalization-deep-dive)
   - [Dropout & Regularization Techniques](#dropout--regularization-techniques)
   - [Data Augmentation (Geometric, Color, Mixup, CutMix)](#data-augmentation-geometric-color-mixup-cutmix)
9. [Advanced Hyperparameter Tuning & Transfer Learning](#9-advanced-hyperparameter-tuning--transfer-learning)
   - [Learning Rate Schedules (Cosine Annealing, Warmup)](#learning-rate-schedules-cosine-annealing-warmup)
   - [Transfer Learning Across Domains](#transfer-learning-across-domains)
10. [Deploying Deep Learning Models](#10-deploying-deep-learning-models)
    - [Model Exports (ONNX, TorchScript, SavedModel, TFLite)](#model-exports-onnx-torchscript-savedmodel-tflite)
    - [Model Optimization (Quantization, Pruning, Distillation)](#model-optimization-quantization-pruning-distillation)
    - [Production Deployment Checklist](#production-deployment-checklist)
11. [Unit III Cheat Sheet & Quick Reference](#11-unit-iii-cheat-sheet--quick-reference)

---

## 1. Generative vs. Discriminative Models

### Analogy: The Art Critic vs. The Painter

Imagine two AI models evaluating artwork:
- **Discriminative Model (The Art Critic):** It takes a completed painting ($x$) and determines whether it was painted by Picasso or Monet ($y$). It doesn't know how to paint; it only learns the **boundary separating classes**.
- **Generative Model (The Painter):** It studies hundreds of Picasso paintings, learns the underlying rules of cubism, and can **paint a brand new Picasso-style portrait from scratch** ($x$).

```
DISCRIMINATIVE MODEL : Inputs (X) -------------> [ Learns P(Y|X) Boundary ] -------------> Output Label (Y)
                                                   "Is this a cat or dog?"

GENERATIVE MODEL     : Random Noise / Class ----> [ Learns P(X) Distribution ] ----------> New Synthetic Image
                                                   "Generate a new cat image!"
```

---

### Joint Probability vs. Conditional Probability

| Aspect | Discriminative Model | Generative Model |
|:---|:---|:---|
| **What it learns** | Conditional probability $P(y|x)$ | Joint probability $P(x, y)$ or input distribution $P(x)$ |
| **Primary Goal** | Classify inputs into categories | Generate new realistic data samples |
| **Examples** | Logistic Regression, CNN, SVM | RBM, VAE, GAN, Diffusion Models |
| **Missing Data Handling** | Poor | Excellent (can marginalize out missing values) |

---

### Energy-Based Models
Many generative models use an **Energy Function** $E(x)$:
$$P(x) = \frac{1}{Z} \exp(-E(x))$$

Where $Z = \sum_x \exp(-E(x))$ is the normalizing **Partition Function**.
- Low Energy = High Probability (Looks like realistic data).
- High Energy = Low Probability (Looks like garbage/noise).

---

## 2. Restricted Boltzmann Machines (RBM)

An **RBM** is a two-layer, undirected generative model consisting of **Visible Units ($v$)** and **Hidden Units ($h$)**.

```
  Hidden Layer (h)   :   ( h1 )       ( h2 )       ( h3 )       ( h4 )
                           \         /  \         /  \         /
                            \       /    \       /    \       /     (No connections WITHIN a layer!)
                             \     /      \     /      \     /
  Visible Layer (v)  :   ( v1 )       ( v2 )       ( v3 )       ( v4 )
```

### Why "Restricted"? (Conditional Independence)
In a standard Boltzmann machine, all neurons connect to each other. In a **Restricted** Boltzmann Machine:
- **No visible-to-visible connections.**
- **No hidden-to-hidden connections.**

Because of this restriction, given the visible input $v$, **all hidden neurons can be calculated in parallel!**
$$P(h_j = 1 | v) = \sigma\left(b_j + \sum_i w_{ij} v_i\right)$$
$$P(v_i = 1 | h) = \sigma\left(a_i + \sum_j w_{ij} h_j\right)$$

---

### Contrastive Divergence (CD-k) Training
Calculating the exact partition function $Z$ requires summing over millions of state combinations (intractable). RBMs use **Contrastive Divergence (CD-1)**:

```
1. Positive Phase : Clamp real data onto Visible (v0)  ---> Sample Hidden (h0)
2. Negative Phase : Reconstruct Visible from h0 (v1)    ---> Sample Hidden (h1)
3. Weight Update  : Weight_new = Weight_old + rate * ( (v0 * h0) - (v1 * h1) )
```
*Intuition: Show the model real data, let it "dream" for 1 step, and adjust weights so reality is more probable than its dream.*

---

## 3. Deep Belief Networks (DBN)

A **Deep Belief Network (DBN)** is created by stacking multiple RBMs on top of each other.

```
+------------------------------------+
|            RBM Layer 3             |
+------------------------------------+
                  ^  (Hidden becomes Visible for next layer)
+------------------------------------+
|            RBM Layer 2             |
+------------------------------------+
                  ^
+------------------------------------+
|            RBM Layer 1             |
+------------------------------------+
                  ^
          Raw Input Data (v)
```

### Greedy Layer-Wise Pretraining
1. Train RBM 1 on raw data until convergence.
2. Freeze RBM 1; pass its hidden outputs as visible inputs to train RBM 2.
3. Repeat layer by layer (**Greedy Unsupervised Pretraining**).
4. Add a final classification layer and fine-tune the entire stack with standard Backpropagation.

---

## 4. Autoencoders

An **Autoencoder** is a neural network trained to copy its input to its output through a narrow **Bottleneck (Latent Space)**.

```
Input X (784 dims) ---> [ ENCODER ] ---> Latent Space Z (32 dims) ---> [ DECODER ] ---> Output X_hat (784 dims)
                                              ^
                                      (Bottleneck Code)
```

- **Encoder Equation:** $z = f(W_e x + b_e)$
- **Decoder Equation:** $\hat{x} = g(W_d z + b_d)$
- **Reconstruction Loss (MSE):** $\mathcal{L} = \|x - \hat{x}\|^2$

---

### Types of Autoencoders

| Autoencoder Type | Key Mechanism | Best Purpose |
|:---|:---|:---|
| **Undercomplete** | Latent dimension $k \ll n$ (smaller than input) | Data Compression / Dimensionality Reduction |
| **Sparse** | Adds L1 penalty to activations (forces most hidden nodes to 0) | Feature Extraction |
| **Denoising** | Adds random noise to input, forces model to reconstruct clean output | Image Denoising / Robust feature learning |
| **Contractive** | Adds penalty on Jacobian matrix derivatives | Makes representations resilient to small input shifts |

---

### Autoencoders vs. PCA (Nonlinear Manifolds)
- **PCA (Principal Component Analysis):** Can only project data onto a **flat linear plane**.
- **Autoencoders:** With non-linear activations (ReLU/Sigmoid), autoencoders can learn **curved nonlinear manifolds** (e.g., Swiss roll data).

---

## 5. Variational Autoencoders (VAE)

### Why Regular Autoencoders Fail at Generation
A regular autoencoder maps each input image to a discrete single point in latent space. If you sample a random point from empty gaps in latent space, the decoder outputs garbage!

### Probabilistic Latent Space ($\mu$ and $\sigma$)
Instead of mapping an input to a single point vector $z$, a VAE encoder outputs **two vectors**:
1. Mean vector ($\mu$)
2. Variance vector ($\sigma^2$)

```
Input X ---> [ ENCODER ] ---> Mean (μ)   ----\
                        ---> Log-Var (σ²) ----+---> Sample z ~ N(μ, σ²) ---> [ DECODER ] ---> Reconstructed X
```

---

### The Reparameterization Trick

Sampling $z$ directly from $\mathcal{N}(\mu, \sigma^2)$ is a random operation. **You CANNOT calculate gradients through a random node during Backprop!**

**The Solution:** Move randomness to an independent noise variable $\epsilon \sim \mathcal{N}(0, 1)$:

$$z = \mu + \sigma \odot \epsilon$$

```
WITHOUT TRICK (Broken Backprop):         WITH REPARAMETERIZATION TRICK (Works!):
  μ ----\                                  μ -----------\
         ---> [ Random Sample ] ---> z                   ---> [ z = μ + σ * ε ] ---> z
  σ ----/       (GRADIENT BLOCKED!)        σ ---- (x) --/
                                                   ^
                                 ε ~ N(0,1) -------+
```

---

### VAE Loss Function

$$\mathcal{L}_{\text{VAE}} = \underbrace{\|x - \hat{x}\|^2}_{\text{Reconstruction Loss (Looks like original)}} + \underbrace{D_{\text{KL}}\left( \mathcal{N}(\mu, \sigma^2) \,||\, \mathcal{N}(0, 1) \right)}_{\text{KL Divergence (Keeps latent space smooth and continuous)}}$$

---

## 6. Reinforcement Learning & Deep Q-Networks (DQN)

In **Reinforcement Learning (RL)**, an **Agent** learns to make decisions by taking **Actions** in an **Environment** to maximize cumulative **Rewards**.

```
                      +-------------------+
                      |    ENVIRONMENT    |
                      +-------------------+
                        /               \
       State (s_t) & Reward (r_t)     Action (a_t)
                      /                   \
                     v                     |
              +---------------+            |
              |     AGENT     | -----------+
              +---------------+
```

---

### Key Terms & Definitions
- **State ($s$):** The current scenario (e.g., screen pixels in a video game).
- **Action ($a$):** The choice made by the agent (e.g., Move Left, Jump, Move Right).
- **Reward ($r$):** Scalar feedback signal (+10 for scoring, -100 for dying).
- **Policy ($\pi$):** The strategy mapping states to actions.
- **Discount Factor ($\gamma \in [0, 1)$):** Determines how much the agent values future rewards vs. immediate rewards.

---

### Q-Learning & The Bellman Equation
The **Q-Value** $Q(s, a)$ measures the expected future return of taking action $a$ in state $s$:

$$Q(s, a) = r + \gamma \max_{a'} Q(s', a')$$

---

### DQN Innovations: Experience Replay & Target Networks

Standard Q-learning with neural networks is unstable. **DQN** introduced two major fixes:

1. **Experience Replay Buffer:**
   - Store past experiences $(s, a, r, s')$ in a memory buffer.
   - Train on random mini-batches sampled from the buffer.
   - **Benefit:** Breaks temporal correlations between consecutive video frames.

2. **Separate Target Network:**
   - Use a second frozen network $Q_{\text{target}}$ to compute target values ($r + \gamma \max Q_{\text{target}}$).
   - Periodically update $Q_{\text{target}}$ weights every $C$ steps.
   - **Benefit:** Stops the target from constantly shifting during updates.

---

## 7. Generative Adversarial Networks (GANs)

Invented by Ian Goodfellow in 2014, **GANs** train two networks simultaneously in a competitive zero-sum game.

### The Counterfeiter vs. Police Game
- **Generator ($G$):** The Counterfeiter. Takes random noise vector $z$ and creates fake images trying to fool the discriminator.
- **Discriminator ($D$):** The Police. Examines real images and fake images and outputs probability $D(x) \in [0, 1]$ ($1 = \text{Real}, 0 = \text{Fake}$).

```
Random Noise (z) ---> [ GENERATOR (G) ] ---> Fake Image \
                                                         +---> [ DISCRIMINATOR (D) ] ---> Real or Fake?
Real Training Dataset ---------------------> Real Image /
```

---

### Minimax Objective Function
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

- **Discriminator wants to MAXIMIZE:** $D(x) \rightarrow 1$ (Real=Real) and $D(G(z)) \rightarrow 0$ (Fake=Fake).
- **Generator wants to MINIMIZE:** Make $D(G(z)) \rightarrow 1$ (Fool Discriminator into calling Fake=Real).

---

### Architectures: DCGAN & Conditional GAN (cGAN)

- **DCGAN (Deep Convolutional GAN):** Replaces dense layers with Transposed Convolutions in $G$ and Strided Convolutions in $D$.
- **Conditional GAN (cGAN):** Feeds class labels $y$ into BOTH $G$ and $D$ so you can control output (e.g., *"Generate a cat"* vs *"Generate a dog"*).

---

### Mode Collapse & Stability Tricks
- **Mode Collapse:** A major GAN failure where $G$ produces only ONE single plausible output over and over (e.g., only generating digit 1s).
- **Fixes:** Use **Wasserstein GAN (WGAN)** loss, Spectral Normalization, or Minibatch Discrimination.

---

## 8. Training Challenges & Solutions

### A. Vanishing & Exploding Gradients
- **Vanishing Gradients:** Gradients shrink near 0; early layers stop learning. (*Fix: Use ReLU, ResNet skip connections, He initialization*).
- **Exploding Gradients:** Gradients become huge numbers; model output becomes `NaN`. (*Fix: Gradient Clipping*).

---

### B. Batch Normalization Deep-Dive
Batch Norm normalizes activations across a mini-batch:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

- $\gamma$ (scale) and $\beta$ (shift) are learnable parameters.
- **Benefits:** Prevents internal covariate shift, allows $10\times$ higher learning rates, acts as regularizer.

---

### C. Dropout & Regularization Techniques

```
    WITHOUT DROPOUT                        WITH DROPOUT (p = 0.5)
  (o)---(o)---(o)                        (o)   (x)   (o)
   | \ / | \ / |                          |     |     |       (50% of neurons randomly deactivated
  (o)---(o)---(o)                        (x)   (o)   (x)       during training pass!)
   | \ / | \ / |                          |     |     |
  (o)---(o)---(o)                        (o)   (x)   (o)
```

- **Inverted Dropout:** Scales remaining active neurons by $\frac{1}{1-p}$ during training so no scaling is needed at test time.

---

### D. Data Augmentation Strategies

```
Original Image        Horizontal Flip         Random Crop          CutMix Blend
  +---------+           +---------+           +---------+           +---------+
  |  (Cat)  |   ===>    |  (taC)  |   ===>    |  (Ca)   |   ===>    | (Cat+Dog|
  +---------+           +---------+           +---------+           +---------+
```
- **Mixup:** Blends two images linearly: $\tilde{x} = \lambda x_1 + (1-\lambda) x_2$.
- **CutMix:** Pastes a patch of Image B onto Image A.

---

## 9. Advanced Hyperparameter Tuning & Transfer Learning

### Learning Rate Schedules

```
     Cosine Annealing Decay                        Warmup + Decay
  LR ^                                         LR ^     / \
     | \                                          |    /   \
     |   \                                        |   /     \
     |     ~__                                    |  /       \___
     +--------------> Epochs                      +--------------> Epochs
```

- **Warmup:** Slowly ramps up learning rate for first few epochs to protect pretrained weights from sudden shock.

---

## 10. Deploying Deep Learning Models

### Model Exports

```
PyTorch (.pt)  -----> ONNX Export (.onnx) -----> TensorRT / TorchScript (Fast Inference)
TensorFlow     -----> SavedModel          -----> TFLite (Mobile / Edge devices)
```

---

### Model Optimization Techniques

```
1. QUANTIZATION  : Converts 32-bit Floating Point (FP32) weights --> 8-bit Integers (INT8).
                   Result: 4x smaller memory size, 3x faster speed!

2. PRUNING       : Removes near-zero weight connections entirely.
                   Result: Sparse matrices, faster execution.

3. DISTILLATION  : Trains a small "Student Model" to copy the probability predictions of a massive "Teacher Model".
```

---

### Production Deployment Checklist
1. ☐ **Export to ONNX / TorchScript / TFLite**.
2. ☐ **Quantize to INT8 / FP16**.
3. ☐ **Profile Inference Latency** (Ensure <50ms response time).
4. ☐ **Set up Serving Framework** (TorchServe, Triton Server, TF Serving).
5. ☐ **Monitor Data Drift & Output Anomalies in Production**.

---

## 11. Unit III Cheat Sheet & Quick Reference

| Model / Concept | Main Equation / Objective | Primary Application |
|:---|:---|:---|
| **RBM** | $E(v,h) = -a^Tv - b^Th - v^TWh$ | Feature extraction, Collaborative filtering |
| **Autoencoder** | $\min \|x - \text{Decoder}(\text{Encoder}(x))\|^2$ | Non-linear dimensionality reduction, Denoising |
| **VAE** | $\text{Reconstruction Loss} + D_{\text{KL}}(\mathcal{N}(\mu,\sigma^2) \| \mathcal{N}(0,1))$ | Smooth generative latent spaces, Interpolation |
| **DQN** | $Q(s,a) \leftarrow r + \gamma \max_{a'} Q_{\text{target}}(s', a')$ | Deep Reinforcement Learning (Atari, Robotics) |
| **GAN** | $\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$ | Photorealistic image synthesis, Style transfer |
| **Quantization** | $\text{FP32} \rightarrow \text{INT8}$ | Ultra-fast edge model deployment |
