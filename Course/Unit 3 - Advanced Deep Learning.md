# Unit III — Advanced Deep Learning

Course: 25CSA543A — Deep Learning for AI

---

## Joint Probability Density Functions and Generative Models

Before diving into RBMs and GANs, we need to understand what makes a model "generative."

### Discriminative vs Generative

- **Discriminative model** — learns the decision boundary directly. Given input $x$, it models $P(y|x)$ — "what class does this belong to?" Examples: logistic regression, standard CNNs, SVMs.
- **Generative model** — learns the underlying data distribution $P(x)$ (or the joint $P(x, y)$). It can then generate new samples that look like the training data. Examples: RBMs, VAEs, GANs.

### Joint probability density function

The joint pdf $P(x, y)$ captures the probability of seeing both a particular input $x$ and label $y$ together. From the joint, you can derive everything:

- **Marginal**: $P(x) = \sum_y P(x, y)$ — probability of the input regardless of class
- **Conditional**: $P(y|x) = P(x, y) / P(x)$ — this is what discriminative models learn directly
- **Bayes' rule**: $P(y|x) = P(x|y) \cdot P(y) / P(x)$ — generative models often use this path

### Why generative models?

| Capability | Discriminative | Generative |
|-----------|---------------|------------|
| Classification | ✅ Direct, often more accurate | ✅ Via Bayes' rule |
| Generate new samples | ❌ | ✅ Sample from $P(x)$ |
| Handle missing data | ❌ | ✅ Marginalize out missing variables |
| Semi-supervised learning | Limited | ✅ Use unlabeled data to learn $P(x)$ |
| Detect anomalies | Indirect | ✅ Low $P(x)$ = anomaly |

### Energy-based models

Many generative models (including RBMs) define probability through an **energy function**:

$$P(x) = \frac{1}{Z} \exp(-E(x))$$

where $Z = \sum_x \exp(-E(x))$ is the partition function (normalizing constant). Low energy = high probability. The model learns by pushing the energy down for training data and up for everything else.

The challenge is that $Z$ is usually intractable to compute — which is why techniques like contrastive divergence (CD) and variational inference exist.

---

## Restricted Boltzmann Machines (RBM)

An RBM is an **energy-based generative model** with two layers — visible units **v** and hidden units **h** — but *no connections within the same layer*. That constraint (no visible-visible or hidden-hidden links) is what makes it "restricted" compared to a full Boltzmann machine, and it's also what makes training tractable.

### How it works

- Each connection between a visible unit $v_i$ and hidden unit $h_j$ has a weight $w_{ij}$
- Each unit also has a bias: $a_i$ for visible, $b_j$ for hidden
- The network defines an **energy function** over every possible configuration of (v, h):

$$E(v, h) = -\sum_i a_i v_i - \sum_j b_j h_j - \sum_{i,j} v_i w_{ij} h_j$$

Lower energy = more probable configuration. The model learns by pushing down the energy of training data configurations and pushing up the energy of everything else.

### Why no intra-layer connections matter

Because there are no connections within a layer, all hidden units are **conditionally independent** given the visible layer (and vice versa). This means:

- Given input v, you can sample all hidden units in parallel: $P(h_j = 1 | v) = \sigma(b_j + \sum_i w_{ij} v_i)$
- Given hidden state h, you can reconstruct all visible units in parallel: $P(v_i = 1 | h) = \sigma(a_i + \sum_j w_{ij} h_j)$

This parallel sampling is what makes RBMs practical — a full Boltzmann machine would need slow iterative Gibbs sampling.

### Contrastive Divergence (CD-k)

Training an RBM requires computing the gradient of the log-likelihood, which involves an intractable partition function. **Contrastive Divergence** is the clever workaround:

1. **Positive phase** — clamp training data on visible layer, sample hidden units
2. **Negative phase** — reconstruct visible units from those hidden units, then re-sample hidden units
3. **Update rule**: $\Delta w_{ij} = \eta \left( \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{recon}} \right)$

The idea: we only run the Gibbs chain for *k* steps (usually k=1) instead of running it to convergence. It's biased but works surprisingly well in practice.

Think of it like this — you show the model real data, let it "dream" one step, and then nudge the weights so reality looks more probable than the dream.

### What RBMs are good at

- Feature extraction from unlabelled data
- Pretraining layers of deep networks (more on this below)
- Collaborative filtering (Netflix Prize famously used RBMs)
- Dimensionality reduction

---

## Deep Belief Networks (DBN)

A DBN is built by **stacking multiple RBMs** on top of each other. The hidden layer of one RBM becomes the visible layer of the next. The key innovation is **greedy layer-wise pretraining**.

### Greedy layer-wise pretraining

Why "greedy"? Because you train one layer at a time, freezing the previous ones:

1. Train the first RBM on raw input data
2. Use the learned hidden representations as input to train the second RBM
3. Repeat for as many layers as you want
4. **Fine-tune** the entire stack with supervised backpropagation (using labels)

Each new layer is guaranteed (in theory) to improve a lower bound on the log-likelihood of the data. In practice, this was a breakthrough — before this technique (Hinton et al., 2006), deep networks were notoriously hard to train because of vanishing gradients.

### Why it helped historically

- Random initialization of deep networks often landed in terrible local minima
- Layer-wise pretraining gave the network a much better starting point
- Each layer learns increasingly abstract features — edges → shapes → objects
- Once modern tricks (ReLU, batch norm, Adam) arrived, pretraining became less critical, but the idea is still foundational

| Aspect | Single RBM | DBN (stacked RBMs) |
|--------|-----------|-------------------|
| Depth | 2 layers (visible + hidden) | Many layers |
| Training | Contrastive divergence | Greedy layer-wise + fine-tune |
| Features | Low-level | Hierarchical (low → high) |
| Use case | Feature extraction | Classification, generation |

---

## Autoencoders

An autoencoder learns to **compress** input into a lower-dimensional representation and then **reconstruct** it. The network is forced to learn the most important features because the bottleneck is smaller than the input.

### Architecture

```
Input (n dims) → Encoder → Latent space (k dims, k << n) → Decoder → Reconstructed input (n dims)
```

- **Encoder**: maps input $x$ to latent code $z = f(Wx + b)$
- **Decoder**: maps latent code back to reconstruction $\hat{x} = g(W'z + b')$
- **Loss**: reconstruction error, typically MSE: $\mathcal{L} = \|x - \hat{x}\|^2$

The encoder and decoder are just neural networks — they can be as simple as a single linear layer or as deep as you want.

### Types of autoencoders

- **Undercomplete** — latent dimension < input dimension (forces compression)
- **Sparse** — adds a sparsity penalty so most latent units are inactive; learns more interesting features even when the latent space is large
- **Denoising** — input is corrupted with noise, but the target is the *clean* version; forces the model to learn robust features rather than just copying
- **Contractive** — adds a penalty on the Jacobian of the encoder, making the representation insensitive to small input perturbations

### What makes autoencoders different from PCA?

PCA finds the best *linear* subspace. Autoencoders with nonlinear activations can capture **curved manifolds** in the data — they're doing nonlinear dimensionality reduction. With linear activations and MSE loss, an autoencoder actually recovers the same subspace as PCA.

---

## Variational Autoencoder (VAE)

Regular autoencoders map each input to a single point in latent space. VAEs instead map each input to a **probability distribution** — specifically a Gaussian with learned mean $\mu$ and variance $\sigma^2$. This turns the autoencoder into a proper generative model.

### The key idea

- Encoder outputs two vectors: $\mu(x)$ and $\log \sigma^2(x)$
- Sample from the latent distribution: $z \sim \mathcal{N}(\mu, \sigma^2)$
- Decoder reconstructs from the sample


### Reparameterization trick

There's a problem — sampling from $\mathcal{N}(\mu, \sigma^2)$ is a **random operation**, and you can't backpropagate through randomness. The reparameterization trick fixes this:

Instead of sampling $z$ directly, sample $\epsilon \sim \mathcal{N}(0, 1)$ and compute:

$$z = \mu + \sigma \cdot \epsilon$$

Now the randomness is in $\epsilon$ (which doesn't depend on any parameters), and the gradients flow through $\mu$ and $\sigma$ just fine.

### VAE loss function

The loss has two parts:

$$\mathcal{L} = \underbrace{\|x - \hat{x}\|^2}_{\text{reconstruction}} + \underbrace{D_{KL}\left(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)\right)}_{\text{regularization}}$$

- **Reconstruction loss** — how well did the decoder reconstruct the input?
- **KL divergence** — how far is the learned distribution from a standard normal?

The KL term prevents the model from collapsing each input to a tiny point (which would defeat the purpose). It keeps the latent space smooth and continuous — neighboring points in latent space produce similar outputs, which means you can **interpolate** between data points and generate new samples by sampling from $\mathcal{N}(0, 1)$.

| Feature | Regular Autoencoder | VAE |
|---------|-------------------|-----|
| Latent space | Deterministic point | Probability distribution |
| Generation | Not straightforward | Sample from $\mathcal{N}(0, 1)$ and decode |
| Loss | Reconstruction only | Reconstruction + KL divergence |
| Latent structure | Irregular, gaps | Smooth, continuous |

---

## Applications of Autoencoders & Generative Models

### Semi-supervised classification

When labeled data is scarce but unlabeled data is abundant:

- Pretrain an autoencoder (or DBN) on all the data — labeled and unlabeled
- The encoder learns useful feature representations without needing labels
- Attach a classifier head to the encoder and fine-tune on the small labeled set
- The pretrained features give the classifier a massive head start

This works because the autoencoder learns the underlying data structure, which is useful regardless of the specific classification task.

### Noise reduction (denoising)

Denoising autoencoders are trained on corrupted inputs but target clean outputs. Once trained:

- Feed a noisy image through the encoder and decoder
- The network strips away noise while preserving signal
- Works for image denoising, audio cleanup, and even filling in missing data

The key insight is that the model learns the **manifold of clean data** — noise pushes data off this manifold, and the autoencoder projects it back.

### Nonlinear dimensionality reduction

- PCA gives you the best linear subspace — but real data often lives on curved manifolds
- Autoencoders with nonlinear activations can learn these curved structures
- The latent space of an autoencoder is a nonlinear embedding of the data
- Useful for visualization (reduce to 2D/3D), feature extraction, and data compression

Example: reducing a 784-dimensional MNIST image to 2 latent dimensions — similar digits cluster together in the latent space, even though no labels were used during training.

---

## Goal-Oriented Decision Making — Reinforcement Learning Basics

Reinforcement learning (RL) is fundamentally different from supervised learning. Instead of learning from labeled examples, an **agent** learns by interacting with an **environment** and receiving **rewards**.

### Core concepts

- **Agent** — the learner/decision-maker
- **Environment** — everything the agent interacts with
- **State** ($s$) — current situation the agent is in
- **Action** ($a$) — what the agent can do
- **Reward** ($r$) — feedback signal after taking an action
- **Policy** ($\pi$) — the agent's strategy: mapping from states to actions
- **Episode** — one complete run from start to terminal state

The goal: find a policy $\pi$ that maximizes the **cumulative discounted reward**:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

The discount factor $\gamma \in [0, 1)$ controls how much the agent cares about future vs. immediate rewards. $\gamma = 0$ makes the agent greedy (only cares about next reward), $\gamma \to 1$ makes it far-sighted.

### Exploration vs. exploitation

The classic RL dilemma:
- **Exploit** — do what you currently think is best
- **Explore** — try something new, might discover something better

Too much exploitation → stuck in suboptimal behavior. Too much exploration → never capitalizes on what it's learned. Every RL algorithm needs to balance these.

---

## Policy and Target Networks

When you combine RL with deep neural networks (Deep RL), training becomes unstable. The issue: the network is trying to hit a moving target — the Q-values it's chasing keep changing as the network updates.

### Why two networks?

- **Policy network** (online network) — the one being actively trained, used to select actions
- **Target network** — a frozen copy of the policy network, used to compute target Q-values

The target network is updated slowly (either periodically copied from the policy network, or via **soft updates**):

$$\theta_{\text{target}} \leftarrow \tau \theta_{\text{policy}} + (1 - \tau) \theta_{\text{target}}$$

where $\tau$ is small (e.g., 0.005). This keeps the target stable enough for learning to converge.

Without the target network, you'd be adjusting your predictions based on predictions from the same rapidly-changing network — like trying to shoot at a moving target that moves every time you aim. The separate target network holds still long enough for you to learn.

---

## Deep Q-Network (DQN)

DQN (Mnih et al., 2015) was the first deep RL method to achieve human-level performance on Atari games. It combines Q-learning with deep neural networks and two critical stabilization tricks.

### Q-learning recap

The **Q-function** $Q(s, a)$ estimates the expected cumulative reward of taking action $a$ in state $s$ and then following the optimal policy. The Bellman equation gives us the update rule:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

In plain English: adjust your current estimate toward the reward you got plus the best you can do from the next state.

With a small discrete state space you can store Q-values in a table. But for complex environments (images, continuous states), you need a neural network to approximate $Q(s, a; \theta)$ — that's DQN.

### Experience replay

Instead of training on experiences in order (which are highly correlated), DQN stores transitions $(s, a, r, s')$ in a **replay buffer** and samples random mini-batches for training.

Why this matters:
- Breaks temporal correlations between consecutive samples
- Each experience can be reused multiple times (data efficient)
- Smooths out the training distribution

Without replay, the network overfits to whatever it's currently experiencing and forgets earlier lessons.

### Epsilon-greedy exploration

DQN uses a simple but effective exploration strategy:

- With probability $\epsilon$, choose a **random** action (explore)
- With probability $1 - \epsilon$, choose the action with highest Q-value (exploit)
- Start with high $\epsilon$ (e.g., 1.0) and **decay** it over training (e.g., down to 0.01)

Early on, the agent explores wildly. As it learns, it increasingly trusts its own Q-value estimates.

### Putting it all together

```
Initialize replay buffer D
Initialize policy network Q with random weights θ
Initialize target network Q̂ with weights θ⁻ = θ

For each episode:
    Observe initial state s
    For each step:
        Select action a (epsilon-greedy from Q)
        Execute a, observe reward r and next state s'
        Store (s, a, r, s') in D
        Sample random mini-batch from D
        Compute target: y = r + γ max_a' Q̂(s', a'; θ⁻)
        Update θ by minimizing (y - Q(s, a; θ))²
        Every C steps: θ⁻ ← θ
```

---

## Generative Adversarial Networks (GANs)

GANs (Goodfellow et al., 2014) train two networks against each other in a game — one generates fake data, the other tries to tell real from fake. Through this competition, the generator learns to produce increasingly realistic outputs.

### The two players

- **Generator** $G(z)$ — takes random noise $z \sim \mathcal{N}(0, 1)$ and produces fake data
- **Discriminator** $D(x)$ — takes data (real or fake) and outputs probability that it's real

### The minimax game

The training objective:

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

Breaking this down:
- The discriminator wants to maximize — correctly classify real as real ($D(x) \to 1$) and fake as fake ($D(G(z)) \to 0$)
- The generator wants to minimize — fool the discriminator ($D(G(z)) \to 1$)

### Training dynamics

Training alternates between updating D and G:

1. **Train D**: show it real samples and fake samples from G, update D to distinguish them better
2. **Train G**: generate fake samples, pass through D, update G to make D's output closer to 1 (i.e., fool D)

The equilibrium (in theory): G produces data indistinguishable from real data, and D outputs 0.5 for everything (can't tell the difference).

In practice, training GANs is notoriously finicky — more on that below.

---

## Generator and Discriminator Architectures

### DCGAN (Deep Convolutional GAN)

DCGAN (Radford et al., 2016) established architectural guidelines that made GAN training more stable:

- **Generator**: uses transposed convolutions (upsampling) to go from noise vector → image
- **Discriminator**: uses strided convolutions (downsampling) to go from image → real/fake probability
- **Key rules**:
  - Replace pooling layers with strided convolutions (discriminator) and transposed convolutions (generator)
  - Use batch normalization in both networks (except output layer of G, input layer of D)
  - Use ReLU in generator (except output: tanh), LeakyReLU in discriminator
  - No fully connected layers (except for input/output)

### Conditional GAN (cGAN)

Regular GANs generate random outputs — you can't control *what* they produce. Conditional GANs fix this by feeding a **condition** (e.g., class label, text description) to both the generator and discriminator:

- Generator: $G(z, c)$ — noise + condition → output
- Discriminator: $D(x, c)$ — data + condition → real/fake

Example: give the generator the label "7" and it produces images of the digit 7.

### Mode collapse

The most common GAN failure mode. The generator finds a few outputs that fool the discriminator and keeps producing only those — ignoring the diversity of the real data.

Signs of mode collapse:
- Generator always produces similar-looking outputs
- Low variety across generated samples
- Discriminator loss oscillates but doesn't improve

Mitigation strategies:
- **Minibatch discrimination** — let D see batches of samples, not just individual ones
- **Unrolled GANs** — G anticipates future D updates
- **Wasserstein GAN (WGAN)** — uses Wasserstein distance instead of JS divergence, provides more stable gradients
- **Spectral normalization** — constrains D's Lipschitz constant

| GAN Variant | Key Feature | Best For |
|-------------|------------|----------|
| DCGAN | Convolutional architecture | Image generation |
| cGAN | Conditioned on labels/attributes | Controlled generation |
| WGAN | Wasserstein distance loss | Stable training |
| CycleGAN | Unpaired image translation | Style transfer |
| StyleGAN | Style-based generator | High-res face synthesis |

---

## Challenges in Neural Network Training

### Vanishing and exploding gradients

In deep networks, gradients are multiplied through many layers during backpropagation. If the weights are slightly less than 1, gradients shrink exponentially (vanish). If slightly greater than 1, they grow exponentially (explode).

- **Vanishing gradients** — early layers barely learn because their gradients are near zero. Common with sigmoid/tanh activations.
- **Exploding gradients** — weights swing wildly, loss becomes NaN. Can happen with poor initialization.

Solutions:
- **ReLU activation** — gradients are either 0 or 1, no multiplicative shrinking
- **Proper initialization** — Xavier init for tanh, He init for ReLU: $w \sim \mathcal{N}(0, \sqrt{2/n_{\text{in}}})$
- **Residual connections** — skip connections let gradients flow directly through (ResNet)
- **Gradient clipping** — cap gradient magnitude to prevent explosions

### Batch normalization

Normalizes the inputs to each layer so they have zero mean and unit variance, then applies learnable scale ($\gamma$) and shift ($\beta$):

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta$$

Why it helps:
- Reduces internal covariate shift — each layer sees more stable inputs
- Allows higher learning rates
- Acts as mild regularization (because batch statistics add noise)
- Makes the network less sensitive to initialization

At test time, use running averages of mean/variance computed during training (not batch statistics).

### Dropout

Randomly sets a fraction of neurons to zero during each training step. This forces the network to not rely on any single neuron and learn more robust, distributed representations.

- Training: each neuron is dropped with probability $p$ (commonly $p = 0.5$ for hidden, $p = 0.2$ for input)
- The remaining activations are scaled by $\frac{1}{1-p}$ to maintain expected values (inverted dropout)
- Test time: all neurons are active, no scaling needed

Dropout is effectively training an ensemble of exponentially many sub-networks that share weights.

### Regularization summary

| Technique | How it works | When to use |
|-----------|-------------|-------------|
| L2 (weight decay) | Penalizes large weights: $\lambda \sum w^2$ | Default regularizer |
| L1 | Penalizes weight magnitude: $\lambda \sum |w|$ | When you want sparse weights |
| Dropout | Randomly drops neurons | Large fully-connected layers |
| Batch norm | Normalizes layer inputs | Almost always (CNNs especially) |
| Early stopping | Stop when val loss stops improving | Always monitor this |
| Data augmentation | Increase effective training set | When data is limited |

---

## Data Augmentation

When you don't have enough training data, you can artificially expand your dataset by applying transformations that preserve the label. The model sees each training example in many variations, which reduces overfitting.

### Geometric transforms

- **Horizontal flip** — works for most natural images (not text or directional data)
- **Random crop** — forces the model to recognize objects at different positions
- **Rotation** — small angles (±15°) for most tasks, full 360° for aerial/medical images
- **Scaling/zoom** — randomly resize to handle scale variation
- **Shear/affine** — slight perspective distortion

### Color and intensity

- **Color jitter** — randomly adjust brightness, contrast, saturation
- **Random grayscale** — occasionally convert to grayscale
- **Normalization** — per-channel mean/std normalization (not augmentation per se, but essential)

### Advanced techniques

- **Mixup** — blend two training images and their labels: $\tilde{x} = \lambda x_1 + (1-\lambda) x_2$, same for labels. Encourages linear behavior between training examples.
- **Cutout** — randomly mask a square patch of the input with zeros. Forces the model to use context, not just key features.
- **CutMix** — replace a patch of one image with a patch from another, blend labels proportionally. Combines benefits of Cutout and Mixup.
- **RandAugment** — randomly select from a pool of augmentations with uniform magnitude. Simple, effective, fewer hyperparameters.

The right augmentation strategy depends on the domain — what transformations preserve meaning? Flipping a cat is fine; flipping a "6" makes it a "9".

---

## Hyperparameter Settings (in context of Deep Learning)

Unit I covered hyperparameter basics. Here we focus on the advanced considerations that arise when training deep networks at scale.

### Key hyperparameters for deep networks

| Hyperparameter | Typical Range | How to tune |
|---------------|--------------|-------------|
| Learning rate | 1e-4 to 1e-1 | Most important — use LR finder (sweep LR over one epoch, pick steepest descent region) |
| Batch size | 16–512 | Larger = faster training, but may generalize worse. Scale LR proportionally (linear scaling rule) |
| Weight decay (L2) | 1e-5 to 1e-2 | Regularization strength — higher = simpler model |
| Dropout rate | 0.1–0.5 | Higher for larger networks or less data |
| Number of layers/units | Architecture-dependent | Start small, increase until dev error stops improving |
| Optimizer params ($\beta_1$, $\beta_2$) | 0.9, 0.999 (Adam defaults) | Rarely need to change from defaults |

### Learning rate scheduling

A fixed learning rate is rarely optimal throughout training:

- **Step decay** — reduce LR by factor (e.g., ×0.1) every N epochs. Simple and effective.
- **Cosine annealing** — smoothly decay LR following a cosine curve. Used in most modern training recipes.
- **Warm-up + decay** — start with tiny LR, ramp up linearly for a few epochs, then decay. Essential for large batch training and Transformers.
- **Reduce on plateau** — monitor dev loss; if it stops improving for N epochs, reduce LR. Good for fine-tuning.

$$\alpha = \frac{1}{1 + \text{decay\_rate} \times \text{epoch}} \cdot \alpha_0$$

### Practical strategies

- **Random search > grid search** — Bergstra & Bengio (2012) showed that random sampling covers the important dimensions better because hyperparameters vary in importance.
- **Sample on log scale** — learning rate and regularization should be sampled uniformly in log space (e.g., $10^{-4}$ to $10^{-1}$), not linearly.
- **Coarse-to-fine** — first do a broad sweep to find the right region, then narrow down with more trials in that region.
- **Bayesian optimization** — model the performance surface with a Gaussian process, then intelligently pick the next hyperparameter to try. Available in AWS SageMaker, Optuna, Weights & Biases.

### Weight initialization revisited

- **Xavier/Glorot** — $W \sim \mathcal{N}(0, 1/n_{\text{in}})$ — preserves variance for sigmoid/tanh
- **He/Kaiming** — $W \sim \mathcal{N}(0, 2/n_{\text{in}})$ — accounts for ReLU zeroing half the outputs

Wrong initialization → vanishing or exploding activations from the very first forward pass. Always match initialization to your activation function.

---

## Transfer Learning (in context of Deep Learning)

Unit II introduced transfer learning for CNNs. Here we cover the broader principle and its application across domains.

### The core idea

Instead of training a model from scratch, start with a model pretrained on a large dataset and adapt it to your (often smaller) task. This works because early layers learn general features (edges, textures, basic patterns) that transfer across tasks.

### Transfer learning strategies

| Strategy | When to use | What to do |
|----------|-------------|------------|
| **Feature extraction** | Small dataset, similar domain | Freeze all pretrained layers, train only the final classifier |
| **Fine-tuning (last layers)** | Moderate dataset | Freeze early layers, fine-tune later layers + classifier |
| **Full fine-tuning** | Large dataset, different domain | Initialize with pretrained weights, train everything with small LR |
| **Domain adaptation** | Target domain distribution differs | Use techniques like adversarial alignment to bridge the gap |

### Practical guidelines

- **Freeze early, unfreeze late** — early layers capture universal features (edges, frequencies), later layers capture task-specific ones. Fine-tune from the top down.
- **Use a smaller learning rate** — pretrained weights are already good; large updates would destroy them. Typical: 10x–100x smaller than training from scratch.
- **Discriminative learning rates** — use even smaller LR for early layers, larger for later layers (fastai popularized this).
- **Data size matters**:
  - Very small (< 1000 samples): feature extraction only, don't fine-tune
  - Medium (1k–10k): fine-tune last few layers
  - Large (10k+): fine-tune everything

### Transfer learning across domains

| Source → Target | Examples | What transfers |
|----------------|----------|---------------|
| ImageNet → Medical imaging | Pretrained ResNet → X-ray classification | Edge/texture detectors |
| ImageNet → Satellite imagery | Pretrained CNN → land use classification | Spatial feature extractors |
| Large text corpus → Specific NLP | BERT → sentiment analysis | Language understanding |
| English NLP → Other languages | mBERT → cross-lingual tasks | Multilingual representations |
| Simulation → Real world | Simulated robot → Physical robot (sim-to-real) | Control policies |

### Connection to other Unit III topics

- **Autoencoders** — pretrained encoder weights can be transferred as feature extractors
- **GANs** — pretrained discriminators can serve as feature extractors; pretrained generators can be fine-tuned for new domains
- **DQN** — pretrained CNN feature extractors (from ImageNet) are often used as the visual backbone in deep RL

---

## Deploying ML Models with PyTorch and TensorFlow

Training a model is only half the job — getting it into production reliably is the other half.

### Model export formats

- **PyTorch** → TorchScript (scripted or traced) or ONNX
- **TensorFlow** → SavedModel or TFLite (for mobile/edge)
- **ONNX** (Open Neural Network Exchange) — framework-agnostic format that lets you train in PyTorch and deploy with TensorRT, ONNX Runtime, etc.

```python
# PyTorch → ONNX export
import torch
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["image"], output_names=["prediction"])

# PyTorch → TorchScript
scripted = torch.jit.script(model)
scripted.save("model.pt")
```

```python
# TensorFlow → SavedModel
model.save("saved_model_dir")

# TensorFlow → TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_dir")
tflite_model = converter.convert()
```

### Serving frameworks

- **TorchServe** — official PyTorch model server. Package model into a `.mar` archive, deploy with REST/gRPC endpoints. Handles batching, versioning, metrics.
- **TensorFlow Serving** — similar for TF models, serves SavedModel format via REST/gRPC.
- **Triton Inference Server** (NVIDIA) — supports PyTorch, TF, ONNX, TensorRT. Good for GPU-heavy deployments.

### Optimization for deployment

| Technique | What it does | Speedup |
|-----------|-------------|---------|
| Quantization | Reduce weights from FP32 → INT8 | 2-4x faster, smaller model |
| Pruning | Remove near-zero weights | Smaller, sometimes faster |
| Knowledge distillation | Train small "student" from large "teacher" | Much smaller model |
| Operator fusion | Combine sequential ops into one kernel | Reduced overhead |
| Dynamic batching | Group incoming requests | Better GPU utilization |

### Practical deployment checklist

- Export and test the model in the target format — verify outputs match training framework
- Profile inference latency and memory footprint
- Set up input validation — reject malformed requests before they hit the model
- Implement model versioning — roll back if a new model degrades quality
- Monitor in production — track latency, error rates, prediction distributions
- Consider A/B testing before fully switching to a new model version

---

*End of Unit III — Advanced Deep Learning*