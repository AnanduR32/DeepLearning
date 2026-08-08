# Unit II — Convolutional Neural Networks & Recurrent Neural Networks

> Course: 25CSA543A — Deep Learning for AI

---

# Part A — Convolutional Neural Networks (CNN)

CNNs are designed to work with spatial data — images, videos, and even time-series. The core insight is that neighbouring data points (pixels close together, or consecutive time steps) are highly correlated, and we can exploit this locality instead of treating every input independently.

## Convolution as an Operator

> Convolution between two functions produces a third function expressing how the shape of one is modified by the other.

- A specialized linear operator — a small weight matrix (typically 3×3 or 5×5) that slides across the input from top-left to bottom-right
- This weight matrix is called a **Kernel** or **Filter**
- The sliding step size is called the **stride**, and we often add **padding** (extra border pixels) to control output dimensions

### How it works

- The kernel overlaps a patch of the input
- Each pixel value in the patch is multiplied by the corresponding kernel weight
- All products are summed, usually with a bias term added
- This single scalar goes into the output **feature map**
- The kernel slides to the next position and repeats

### Output size formula

Given:
- Input size: $N \times N$
- Kernel size: $F \times F$
- Padding: $P$
- Stride: $S$

$$\text{Output dimension} = \frac{N - F + 2P}{S} + 1$$

For example, a 32×32 input with a 5×5 kernel, padding 2, stride 1 gives: (32 - 5 + 4)/1 + 1 = **32×32** — same spatial size as input (this is "same" padding).

### Common kernels

| Kernel | Purpose | Matrix |
|--------|---------|--------|
| Identity | No change | Center = 1, rest = 0 |
| Gaussian Blur | Smooth / denoise | Weighted average favoring center |
| Sharpen | Amplify edges | Center > 0, neighbors negative |
| Edge Detection | Find boundaries | Center = 8, all neighbors = -1 |

A 3×3 edge detection kernel:

$$\begin{bmatrix} -1 & -1 & -1 \\ -1 & 8 & -1 \\ -1 & -1 & -1 \end{bmatrix}$$

## Basic CNN Concepts

### Sparse Connectivity

In a fully connected (dense) network, a 1000×1000 image would be flattened to a 1,000,000-length vector — every pixel connects to every neuron in the next layer. That's a massive number of parameters.

In a CNN, each neuron in the next layer connects to only a small local patch (say 3×3 = 9 pixels). This dramatically reduces parameters and makes the network focus on **local patterns** rather than trying to learn global pixel-to-pixel mappings.

### Weight Sharing

- In dense networks: every connection has its own unique weight. A 1000×1000 input to 500 hidden nodes = 500 million weights.
- In CNNs: a single 3×3 kernel (just 9 weights) is reused across the entire image. The same filter slides everywhere, looking for the same pattern regardless of position.

This is why CNNs are **translation invariant** — a cat in the top-left corner looks the same to the network as a cat in the bottom-right.

### Pooling

Pooling layers reduce spatial dimensions to speed up computation and build some position invariance.

- **Max Pooling**: takes the maximum value from each patch (most common, typically 2×2 with stride 2)
- **Average Pooling**: takes the mean value from each patch

A 2×2 max pool with stride 2 halves both width and height — a 224×224 feature map becomes 112×112.

## CNN Architecture — The Building Blocks

A typical CNN stacks three types of layers:

1. **Convolution layers** — extract features using learnable kernels
   - Apply K kernels → get K feature maps → stack them as a volume
   - Kernel depth always matches input depth (3 for RGB input, K for subsequent layers)
   - Early layers detect edges and simple textures
   - Deeper layers detect complex patterns (eyes, wheels, text)

2. **Pooling layers** — downsample to reduce spatial size
   - No learnable parameters — just a fixed operation
   - Provides slight translation invariance

3. **Fully Connected (Dense) layers** — flatten the feature maps and classify
   - Final layer typically uses softmax for multi-class classification

### Receptive Field

Each neuron in a deeper layer "sees" a larger region of the original input. A neuron in layer 1 sees a 3×3 patch. A neuron in layer 2 (after another 3×3 conv) effectively sees a 5×5 patch of the original image. This **receptive field** grows with depth, allowing deeper layers to capture larger-scale features.

### Feature Hierarchy

| Layer Depth | What it learns | Example (face recognition) |
|------------|----------------|---------------------------|
| Early (1-2) | Edges, gradients, colors | Horizontal/vertical edges |
| Middle (3-4) | Textures, parts | Eye shape, nose bridge |
| Deep (5+) | Object parts, whole objects | Full face, pose |

This hierarchical feature learning is what makes CNNs so powerful — they automatically learn features at multiple scales of abstraction.

### Convolution Layer Parameters

- **Number of filters (K)**: how many different features to detect at this layer
- **Filter size (F)**: spatial extent of each kernel (3×3, 5×5, 7×7)
- **Stride (S)**: step size when sliding the kernel
- **Padding (P)**: border pixels added to preserve spatial dimensions
- **Activation**: typically ReLU after each convolution

> Rule of thumb: use small filters (3×3) with more layers rather than large filters with fewer layers. Two 3×3 convolutions have the same receptive field as one 5×5 but with fewer parameters and more non-linearity.


## Popular CNN Architectures

The evolution of CNN architectures is basically a story of going deeper and smarter.

### LeNet-5 (1998) — Where it all started

- Yann LeCun's pioneering architecture for handwritten digit recognition (MNIST)
- 5 layers: Conv → Pool → Conv → Pool → FC → FC → Output
- Used average pooling and tanh/sigmoid activations
- ~60K parameters — tiny by today's standards
- Proved that CNNs could learn useful features automatically

### AlexNet (2012) — The deep learning breakthrough

- Won ImageNet 2012 by a massive margin, sparking the deep learning revolution
- Key innovations over LeNet:
  - Much deeper (8 layers), trained on GPUs for the first time
  - **ReLU activation** instead of tanh — trains much faster
  - **Dropout** for regularization
  - **Data augmentation** (flipping, cropping, color jittering)
  - Local Response Normalization (LRN)
- ~60M parameters, input size 227×227×3

### VGGNet (2014) — Simplicity and depth

- Core idea: use only 3×3 convolutions everywhere, just stack more of them
- VGG-16 (16 weight layers) and VGG-19 (19 layers) are the common variants
- Showed that **depth matters** — going deeper improved accuracy
- ~138M parameters — very heavy, but the architecture is clean and uniform
- Still widely used as a feature extractor in transfer learning

### GoogLeNet / Inception (2014) — Going wider, not just deeper

- Introduced the **Inception module**: instead of choosing one filter size, use 1×1, 3×3, and 5×5 convolutions in parallel and concatenate outputs
- 1×1 convolutions act as dimensionality reduction (bottleneck) before expensive 3×3 and 5×5 ops
- 22 layers deep but only ~5M parameters (much more efficient than VGG)
- Auxiliary classifiers at intermediate layers to help gradient flow during training

### ResNet (2015) — The skip connection revolution

- Solved the degradation problem: very deep networks (>20 layers) were actually performing *worse* than shallower ones, not because of overfitting but because gradients couldn't flow back effectively
- **Residual connections** (skip connections): instead of learning $H(x)$, learn $F(x) = H(x) - x$, then output $F(x) + x$
- The identity shortcut lets gradients flow directly through, enabling networks with 50, 101, even 152 layers
- Key insight: it's easier to learn small residual corrections than entire mappings from scratch
- Won ImageNet 2015, and residual connections are now used everywhere

### Architecture Evolution — Quick Comparison

| Architecture | Year | Depth | Parameters | Key Innovation |
|-------------|------|-------|------------|----------------|
| LeNet-5 | 1998 | 5 | 60K | First practical CNN |
| AlexNet | 2012 | 8 | 60M | ReLU, dropout, GPU training |
| VGGNet | 2014 | 16/19 | 138M | Uniform 3×3 filters, depth |
| GoogLeNet | 2014 | 22 | 5M | Inception module, 1×1 conv |
| ResNet | 2015 | 50-152 | 25-60M | Skip connections |

The trend is clear: architectures got deeper, more parameter-efficient, and introduced clever connectivity patterns to help gradients flow.

## Transfer Learning

Training a deep CNN from scratch requires huge datasets (millions of labeled images) and days of GPU time. Transfer learning lets you leverage models already trained on large datasets like ImageNet.

### The idea

A CNN trained on ImageNet has already learned to detect edges, textures, patterns, and object parts in its lower layers. These features are **generic** — useful for almost any vision task. Only the final layers are specific to ImageNet's 1000 classes.

### How to use it

Three common strategies, depending on how much data you have:

1. **Feature extraction** (very little data)
   - Take a pretrained model (e.g., ResNet-50), remove the final classification layer
   - Freeze all weights — the CNN becomes a fixed feature extractor
   - Add your own classifier (a few dense layers) on top and train only that
   - Works surprisingly well even with a few hundred images

2. **Fine-tuning** (moderate data)
   - Start with pretrained weights, replace the final layer
   - Freeze early layers (they learn generic features), unfreeze later layers
   - Train with a small learning rate so you don't destroy the pretrained features
   - Gradually unfreeze more layers as needed

3. **Full retraining** (lots of data, very different domain)
   - Use pretrained weights as initialization only
   - Retrain the entire network on your dataset
   - Still faster to converge than random initialization

### When to use what

| Your data | Similar to ImageNet? | Strategy |
|-----------|---------------------|----------|
| Small | Yes | Feature extraction |
| Small | No | Fine-tune carefully |
| Large | Yes | Fine-tune aggressively |
| Large | No | Train from scratch (or fine-tune all) |

Transfer learning is also valuable beyond vision — in NLP, pretrained language models (BERT, GPT) follow the same principle.


## Object Detection and Localization

Image classification tells you *what's* in the image. Object detection tells you *what* and *where* — potentially for multiple objects.

### Classification vs Localization vs Detection

- **Classification**: Is there a cat? → Yes/No (or probability per class)
- **Localization**: Where is the cat? → Bounding box coordinates (x, y, width, height)
- **Detection**: Find *all* objects and their locations → Multiple bounding boxes + class labels

### R-CNN Family

**R-CNN** (Regions with CNN features, 2014):
- Use a region proposal algorithm (Selective Search) to suggest ~2000 candidate regions
- Crop and resize each region, run through a CNN to extract features
- Classify each region with SVMs
- Problem: very slow — runs the CNN 2000 times per image

**Fast R-CNN** (2015):
- Run the CNN once on the entire image to get a feature map
- Project proposed regions onto this shared feature map (RoI pooling)
- Much faster, but region proposals still slow

**Faster R-CNN** (2015):
- Replace Selective Search with a **Region Proposal Network (RPN)** — a small CNN that proposes regions
- End-to-end trainable, near real-time

### YOLO (You Only Look Once)

A completely different approach — treats detection as a single regression problem.

- Divides the image into an S×S grid
- Each grid cell predicts B bounding boxes + confidence scores + class probabilities
- Single forward pass through the network → all detections at once
- Extremely fast (real-time detection), trades some accuracy for speed
- YOLO versions have progressively improved accuracy while maintaining speed

| Method | Speed | Accuracy | Key Trade-off |
|--------|-------|----------|---------------|
| R-CNN | Very slow | Good | Runs CNN per region |
| Fast R-CNN | Moderate | Good | Shared features |
| Faster R-CNN | Near real-time | Very good | Learned proposals |
| YOLO | Real-time | Good (improving) | Single-pass detection |

## Face Recognition

### Verification vs Recognition

- **Face Verification** (1:1): "Is this person who they claim to be?" — binary decision, must be highly accurate
- **Face Recognition** (1:K): "Who is this person?" — look up against a database of K known faces, harder problem

### How it works

The network learns an **embedding** — a compact vector representation (typically 128-d) of each face, such that:
- Same person's faces → vectors close together
- Different people's faces → vectors far apart

### Triplet Loss

To train the embedding, use triplets of images:
- **Anchor (A)**: reference image of a person
- **Positive (P)**: another image of the *same* person
- **Negative (N)**: image of a *different* person

The loss pushes the network to satisfy:

$$\|f(A) - f(P)\|^2 + \alpha \leq \|f(A) - f(N)\|^2$$

where $\alpha$ is a margin that enforces a gap between positive and negative pairs. If the anchor-positive distance plus the margin is still less than the anchor-negative distance, the triplet is satisfied.

Choosing hard triplets (where the negative is close to the anchor) is crucial for good training — easy triplets contribute zero gradient.

## Neural Style Transfer

A creative application — take the **content** of one image and render it in the **style** of another.

### How it works

Use a pretrained CNN (typically VGG-19) and define two loss functions:

- **Content loss**: measures how different the generated image's feature activations (at a deep layer) are from the content image's activations
- **Style loss**: measures how different the generated image's feature *correlations* (Gram matrices across channels at multiple layers) are from the style image's correlations

The total loss combines both:

$$L_{total} = \alpha \cdot L_{content} + \beta \cdot L_{style}$$

Then, instead of updating network weights, we **optimize the pixel values** of the generated image via gradient descent to minimize this loss. The ratio $\alpha / \beta$ controls how much content vs style influence the result.

---

# Part B — Recurrent Neural Networks (RNN)

Sequential data is everywhere — text, speech, time series, video, DNA sequences. Traditional feedforward networks treat each input as independent, but in sequential data, **order matters** and each element depends on what came before.

## Recurrent Neural Networks — The Basics

The key difference from feedforward networks: RNNs have a **feedback loop**. Instead of only passing data forward, each hidden unit receives both the current input *and* its own output from the previous time step.

- At each time step $t$, the network takes input $x_t$ and the previous hidden state $h_{t-1}$
- It produces a new hidden state $h_t$ and (optionally) an output $y_t$
- The same weights are shared across all time steps

### The hidden state equation

$$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b)$$

The hidden state $h_t$ is a **summary vector** — it compresses everything the network has seen so far into a fixed-size representation. Early inputs get progressively diluted as new inputs arrive.

### Why RNNs?

- Handle **variable-length** sequences (sentences of any length, time series of any duration)
- **Parameter sharing** across time steps — the same weights process each element, so the model generalizes across positions
- Capture **temporal dependencies** — output depends on the full history, not just the current input

### RNN vs DNN/CNN

| Aspect | DNN/CNN | RNN |
|--------|---------|-----|
| Input | Fixed-size, static | Variable-length sequences |
| Processing | All at once | One element at a time |
| Memory | None | Hidden state carries history |
| Weight sharing | Across space (CNN) | Across time |
| Typical use | Images, tabular data | Text, speech, time series |

## Backpropagation Through Time (BPTT)

Training an RNN means unrolling it across all time steps and computing gradients back through time.

The total loss is the sum of losses at each time step:

$$L = \sum_{t=1}^{T} L_t$$

The gradient of loss w.r.t. weight $W$ requires chaining derivatives back through the hidden states:

$$\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \sum_{k=1}^{t} \frac{\partial L_t}{\partial h_t} \cdot \left(\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}\right) \cdot \frac{\partial h_k}{\partial W}$$

That product $\prod$ of Jacobians is the problem — it involves multiplying the same weight matrix $W$ many times.

## The Vanishing / Exploding Gradient Problem

- If the eigenvalues of $W$ are < 1: gradients **vanish** exponentially — the network can't learn long-range dependencies (forgets early inputs)
- If the eigenvalues are > 1: gradients **explode** — training becomes unstable with massive weight updates

In practice, vanilla RNNs struggle with sequences longer than ~10-20 time steps. Information from early in the sequence gets washed out before it can influence later predictions.

**Gradient clipping** can fix exploding gradients (just cap the gradient norm), but vanishing gradients need architectural solutions — which brings us to LSTM and GRU.


---

## GRU (Gated Recurrent Unit)

GRUs are a simpler alternative to LSTMs. They use gates to control long-term memory but don't need a separate cell state — making them faster to train and requiring fewer parameters.

### Two gates, that's it

- **Update gate** $z_t$ — decides how much of the past to keep vs how much new info to let in
  $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

- **Reset gate** $r_t$ — decides how much of the past to forget when computing the new candidate
  $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

### How it computes the next hidden state

1. **Candidate hidden state** — compute what the new state *could* be, using the reset gate to selectively forget:
   $$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$$

2. **Final state** — blend the old and new using the update gate (linear interpolation):
   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

When $z_t \approx 0$: keep the old state (memory is preserved).
When $z_t \approx 1$: replace with the candidate (new info takes over).

### GRU vs LSTM

| Feature | GRU | LSTM |
|---------|-----|------|
| Gates | 2 (update, reset) | 3 (forget, input, output) |
| Cell state | No (uses hidden state only) | Yes (separate cell + hidden) |
| Parameters | Fewer | More |
| Training speed | Faster per epoch | Slower |
| Performance | Comparable on many tasks | Slightly better on very long sequences |

Use GRUs as the default — switch to LSTMs if you're dealing with very long dependencies and have enough data.

---

## LSTM (Long Short-Term Memory)

LSTMs solve the vanishing gradient problem by introducing a **cell state** $c_t$ — a dedicated memory track that runs through the entire sequence, carrying information forward with minimal transformation.

### Three key concepts

- **Selectively Writing** information to memory
- **Selectively Reading** only what's needed from memory
- **Selectively Forgetting** what's no longer useful

These are implemented using **gates** — each gate is a sigmoid layer (output 0–1) that acts as a valve controlling information flow.

### The three gates

**Forget gate** — what to erase from long-term memory:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Think of it like L1 regularization but in the forward pass — it maintains context purity. If you're reading about motorcycles and a new topic appears, the forget gate can clear the old context.

**Input gate** — what new information to store:

First, decide *what's worth saving*:
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

Then create a *candidate* of new information:
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

**Output gate** — what to output as the hidden state:
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

### Cell state update

The cell state update is the heart of LSTM — it's **additive**, not multiplicative:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

This is why gradients don't vanish — addition doesn't cause the repeated multiplication problem that plagues vanilla RNNs. The gradient can flow backward through the cell state largely unmodified.

The hidden state (actual output) is:
$$h_t = o_t \odot \tanh(c_t)$$

### Worked example intuition

Imagine processing: "The **cat**, which was sitting on the mat and watching birds through the window, **was** sleeping."

- The forget gate keeps "cat" (singular noun) in memory through the long clause
- The input gate doesn't overwrite the subject despite all the intervening words
- When the model reaches "was", the output gate reads the stored subject to correctly predict singular verb form

---

## Encoder-Decoder Models

The encoder-decoder (seq2seq) architecture handles **variable-length input → variable-length output** tasks like translation.

### Architecture

- **Encoder** — processes the entire input sequence and compresses it into a fixed-length **context vector** (the final hidden state)
- **Decoder** — takes that context vector and generates the output sequence, one token at a time

For machine translation ("I love cats" → "J'aime les chats"):

1. Encoder reads "I", "love", "cats" → produces context vector $c$
2. Decoder starts with $c$, generates "J'aime", then "les", then "chats", then STOP token

### The information bottleneck

The core weakness: everything about the input must be squeezed into a single fixed-size vector $c$. For short sentences this works, but for long inputs, the context vector can't possibly capture all relevant information.

This bottleneck is what motivated **attention mechanisms**.

### Training: teacher forcing

During training, we feed the *correct* previous token as input to the decoder (rather than its own prediction). This speeds up convergence but can cause a train-test mismatch called **exposure bias** — the model never learns to recover from its own mistakes.

---

## Attention Mechanism

Attention solves the bottleneck problem by letting the decoder look at **all encoder hidden states**, not just the final one.

### The core idea

Instead of a single context vector, attention computes a *weighted combination* of all encoder states for each decoder step. The weights tell the decoder which parts of the input to focus on.

For each decoder time step $t$:

1. **Score** each encoder hidden state $h_s$ against the current decoder state $d_t$
2. **Normalize** the scores via softmax to get attention weights $\alpha_{t,s}$
3. **Weighted sum** gives the context vector for this step: $c_t = \sum_s \alpha_{t,s} \cdot h_s$

### Bahdanau Attention (additive)

The score function is a small neural network:
$$\text{score}(d_t, h_s) = v^T \cdot \tanh(W_1 d_t + W_2 h_s)$$

This learns which encoder positions are relevant to each decoder step. For translation, the attention weights often show a roughly diagonal pattern — word 3 in the output attends mostly to word 3 in the input (with some reordering for different language word orders).

### Self-Attention (Transformers preview)

Self-attention applies the same idea but within a single sequence — each position attends to all other positions. This is the foundation of Transformers (BERT, GPT).

Each position computes a **Query**, **Key**, and **Value** from its embedding:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The $\sqrt{d_k}$ scaling prevents the dot products from growing too large (which would push softmax into saturation where gradients are tiny).

### Why attention matters

| Without attention | With attention |
|-------------------|----------------|
| Fixed-size bottleneck | Dynamic context per step |
| Struggles with long sequences | Handles long sequences well |
| No interpretability | Attention weights show what model focuses on |
| O(1) memory of input | O(n) access to all positions |

---

## NLP and Word Embeddings

Natural language can't be fed directly into neural networks — we need to convert words to vectors of real numbers.

### One-hot encoding (the naive approach)

Represent each word as a vector with a 1 at its vocabulary index and 0s elsewhere:
- "cat" → [0, 0, 1, 0, ..., 0]
- "dog" → [0, 0, 0, 1, ..., 0]

Problem: every word is equidistant from every other word. "cat" is as similar to "dog" as it is to "democracy." Also, these vectors are enormous (vocab size = 50k+).

### Word2Vec — Learning meaningful representations

Word2Vec learns dense, low-dimensional vectors (typically 100–300 dimensions) where **similar words end up close together**.

**Skip-gram model**: given a center word, predict the surrounding context words.
- Training pair: ("cat", "sat") from "the cat sat on the mat"
- The model learns that "cat" and "dog" appear in similar contexts → similar vectors

**CBOW (Continuous Bag of Words)**: given context words, predict the center word. Faster to train but skip-gram handles rare words better.

Key insight: the learned vectors capture semantic relationships as **vector arithmetic**:
$$\text{king} - \text{man} + \text{woman} \approx \text{queen}$$

### GloVe (Global Vectors)

Instead of predicting context words, GloVe works with the global **co-occurrence matrix** — how often word pairs appear near each other across the entire corpus.

The loss function directly learns vectors such that their dot product approximates the log of their co-occurrence count:
$$w_i^T w_j + b_i + b_j = \log(X_{ij})$$

GloVe tends to perform similarly to Word2Vec but can be more efficient since it works with corpus statistics rather than individual windows.

### Worked example — tokens, vocabulary, and embeddings (step by step)

Let's trace the full pipeline from raw text to neural network input with actual numbers.

**Sentence:** `"the cat sat on the mat"`

**Step 1 — Tokenization** (split into tokens):

| Position | 0 | 1 | 2 | 3 | 4 | 5 |
|----------|---|---|---|---|---|---|
| Token | the | cat | sat | on | the | mat |

Note: "the" appears twice — same token, same vocabulary ID.

**Step 2 — Build vocabulary** (unique tokens → integer IDs):

| Word | Vocab ID |
|------|----------|
| the | 0 |
| cat | 1 |
| sat | 2 |
| on | 3 |
| mat | 4 |

Vocabulary size $V = 5$.

**Step 3 — One-hot encoding** (each word → a $V$-dimensional vector):

$$\text{the} = \begin{bmatrix} 1\\0\\0\\0\\0 \end{bmatrix}, \quad \text{cat} = \begin{bmatrix} 0\\1\\0\\0\\0 \end{bmatrix}, \quad \text{sat} = \begin{bmatrix} 0\\0\\1\\0\\0 \end{bmatrix}, \quad \text{on} = \begin{bmatrix} 0\\0\\0\\1\\0 \end{bmatrix}, \quad \text{mat} = \begin{bmatrix} 0\\0\\0\\0\\1 \end{bmatrix}$$

Problem: these are sparse, high-dimensional ($V$ could be 50,000+), and every pair has the same distance — "cat" is as far from "mat" as it is from "on."

**Step 4 — Embedding lookup** (one-hot × embedding matrix = dense vector):

Suppose we learn a $5 \times 3$ embedding matrix $E$ (5 words, 3-dimensional embeddings):

$$E = \begin{bmatrix} 0.2 & 0.8 & -0.1 \\ 0.9 & 0.3 & 0.5 \\ -0.4 & 0.6 & 0.7 \\ 0.1 & -0.2 & 0.3 \\ 0.8 & 0.4 & 0.6 \end{bmatrix} \leftarrow \text{rows: the, cat, sat, on, mat}$$

To get the embedding for "cat" (ID = 1): just take row 1 of $E$:

$$\text{embed("cat")} = E[1] = \begin{bmatrix} 0.9 & 0.3 & 0.5 \end{bmatrix}$$

Equivalently, $E^T \cdot \text{one\_hot("cat")} = \begin{bmatrix} 0.9 \\ 0.3 \\ 0.5 \end{bmatrix}$ — the matrix multiply just selects row 1.

The full sentence becomes a $6 \times 3$ matrix (6 tokens, 3 dimensions each):

$$\begin{bmatrix} 0.2 & 0.8 & -0.1 \\ 0.9 & 0.3 & 0.5 \\ -0.4 & 0.6 & 0.7 \\ 0.1 & -0.2 & 0.3 \\ 0.2 & 0.8 & -0.1 \\ 0.8 & 0.4 & 0.6 \end{bmatrix} \leftarrow \text{the, cat, sat, on, the, mat}$$

Notice "the" (rows 0 and 4) maps to the same vector — same word, same embedding.

### Worked example — Skip-gram training pairs

**Window size = 1** (look 1 word left and right of the center word):

| Center word | Context words | Training pairs |
|-------------|---------------|----------------|
| the (pos 0) | cat | (the, cat) |
| cat (pos 1) | the, sat | (cat, the), (cat, sat) |
| sat (pos 2) | cat, on | (sat, cat), (sat, on) |
| on (pos 3) | sat, the | (on, sat), (on, the) |
| the (pos 4) | on, mat | (the, on), (the, mat) |
| mat (pos 5) | the | (mat, the) |

With **window = 2**, "sat" at position 2 would produce pairs: (sat, the), (sat, cat), (sat, on), (sat, the) — looking 2 left and 2 right.

**What the model learns from these pairs:**
- "cat" and "mat" both appear next to "the" and near "sat" → they get similar embeddings
- "sat" and "on" share context → similar embeddings
- After training on a large corpus, you'd find: $\text{embed("cat")} \approx \text{embed("dog")}$ because they appear in the same contexts ("the ___ sat", "the ___ ran", etc.)

### Worked example — cosine similarity check

After training, we can verify embeddings make sense. Say we get:

$$\text{cat} = \begin{bmatrix} 0.9 \\ 0.3 \\ 0.5 \end{bmatrix}, \quad \text{mat} = \begin{bmatrix} 0.8 \\ 0.4 \\ 0.6 \end{bmatrix}, \quad \text{on} = \begin{bmatrix} 0.1 \\ -0.2 \\ 0.3 \end{bmatrix}$$

$$\cos(\text{cat}, \text{mat}) = \frac{0.9(0.8) + 0.3(0.4) + 0.5(0.6)}{\sqrt{0.81+0.09+0.25} \times \sqrt{0.64+0.16+0.36}} = \frac{1.14}{1.072 \times 1.077} = \frac{1.14}{1.155} = 0.987$$

$$\cos(\text{cat}, \text{on}) = \frac{0.9(0.1) + 0.3(-0.2) + 0.5(0.3)}{1.072 \times 0.374} = \frac{0.18}{0.401} = 0.449$$

"cat" and "mat" are **much** more similar (0.987) than "cat" and "on" (0.449) — embeddings learned meaningful relationships.

> **Exam tip:** know how to generate skip-gram pairs from a sentence given a window size, compute embedding lookups from the matrix, and verify similarity with cosine distance.

### Transfer learning with embeddings

Pre-trained embeddings (Word2Vec, GloVe, or contextual embeddings like ELMo/BERT) are incredibly valuable:
- Download embeddings trained on billions of words (Google News, Common Crawl)
- Use them as the initial embedding layer in your model
- Fine-tune on your task — especially helpful when you have limited labeled data

---

## Applications

### Sentinel classification

Satellite imagery classification using sequences of multi-spectral images over time. RNNs/LSTMs can model temporal patterns — crop growth cycles, seasonal changes, land use transitions.

The typical pipeline:
- CNN extracts spatial features from each time step's image
- RNN/LSTM processes the sequence of features to capture temporal dynamics
- Final classification: crop type, land cover, or change detection

### Speech recognition

Converting audio waveforms into text using deep learning:

1. **Feature extraction**: raw audio → mel spectrograms or MFCCs (compact frequency-domain representation)
2. **Acoustic model**: RNN/LSTM/Transformer processes the spectrogram sequence
3. **CTC (Connectionist Temporal Classification)**: handles the alignment problem — we don't know which audio frames correspond to which characters. CTC sums over all valid alignments using dynamic programming
4. **Language model**: rescores the output to prefer grammatically correct sentences

Modern end-to-end systems (like Whisper) combine all steps into a single Transformer-based model.

### Action recognition

Recognizing human activities in video sequences:

- **Two-stream approach**: spatial CNN for appearance + temporal CNN for optical flow → fuse predictions
- **3D CNNs** (C3D, I3D): extend 2D convolutions to include the time dimension — kernels slide across both space and time
- **RNN on CNN features**: extract per-frame features with a CNN, feed the sequence to an LSTM for temporal modeling

These approaches combine the spatial understanding of CNNs with the temporal modeling of RNNs.

---

## Quick Reference — Unit II

| Topic | Key Idea | Core Formula / Concept |
|-------|----------|----------------------|
| Convolution | Learnable filter slides over input | Output = $(N-F+2P)/S + 1$ |
| CNN properties | Sparse connectivity + weight sharing | Dramatic parameter reduction |
| Pooling | Downsample feature maps | Max/average over patches |
| LeNet → ResNet | Increasing depth + skip connections | Residual: $F(x) + x$ |
| Transfer learning | Reuse pretrained features | Freeze early layers, fine-tune last |
| Object detection | Locate + classify objects | R-CNN family, YOLO |
| Face recognition | Embed faces → compare distances | Triplet loss |
| Style transfer | Separate content from style | Gram matrix for style |
| RNN | Hidden state loops back | $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t)$ |
| GRU | 2 gates, no cell state | Update + reset gates |
| LSTM | 3 gates + cell state | Additive cell update (no vanishing gradient) |
| Seq2seq | Encoder → context → decoder | Variable input/output length |
| Attention | Weighted sum of all encoder states | Score → softmax → context |
| Word embeddings | Dense word vectors | king - man + woman ≈ queen |
| CTC | Align audio to text | Sum over all valid alignments |

---

*End of Unit II — CNNs and RNNs*