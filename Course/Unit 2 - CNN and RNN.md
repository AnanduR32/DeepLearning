# Unit II — Convolutional Neural Networks (CNN) & Recurrent Neural Networks (RNN)

> **Course Code:** 25CSA543A — Deep Learning for AI  
> **Target Audience:** College Freshmen / Beginners in AI  
> **Core Objective:** Understand how spatial models (CNNs) process images and visual data, and how sequential models (RNNs, LSTMs, GRUs) process text, time-series, and audio.

---

## Table of Contents
1. [PART A: Convolutional Neural Networks (CNN)](#part-a-convolutional-neural-networks-cnn)
   - [What is Convolution? (The Sliding Window Analogy)](#1-what-is-convolution-the-sliding-window-analogy)
   - [Convolution Output Dimension Formula & Examples](#convolution-output-dimension-formula--examples)
   - [Common 3x3 Kernels](#common-3x3-kernels)
   - [Core CNN Concepts: Sparse Connectivity, Weight Sharing, & Pooling](#2-core-cnn-concepts-sparse-connectivity-weight-sharing--pooling)
   - [CNN Architecture & Hierarchy of Features](#3-cnn-architecture--hierarchy-of-features)
   - [Evolution of CNN Architectures (LeNet to ResNet)](#4-evolution-of-cnn-architectures-lenet-to-resnet)
   - [Transfer Learning Strategies](#5-transfer-learning-strategies)
   - [Object Detection & Localization (R-CNN vs. YOLO)](#6-object-detection--localization-r-cnn-vs-yolo)
   - [Face Recognition & Triplet Loss](#7-face-recognition--triplet-loss)
   - [Neural Style Transfer](#8-neural-style-transfer)
2. [PART B: Recurrent Neural Networks (RNN)](#part-b-recurrent-neural-networks-rnn)
   - [Introduction to Sequential Data](#9-introduction-to-sequential-data)
   - [RNN Architecture & The Memory Hidden State](#10-rnn-architecture--the-memory-hidden-state)
   - [Backpropagation Through Time (BPTT) & Gradient Problems](#11-backpropagation-through-time-bptt--gradient-problems)
   - [Gated Recurrent Units (GRU)](#12-gated-recurrent-units-gru)
   - [Long Short-Term Memory Networks (LSTM)](#13-long-short-term-memory-networks-lstm)
   - [Encoder-Decoder (Seq2Seq) Models](#14-encoder-decoder-seq2seq-models)
   - [Attention Mechanism & Self-Attention (Transformers Preview)](#15-attention-mechanism--self-attention-transformers-preview)
   - [Natural Language Processing (NLP) & Word Embeddings](#16-natural-language-processing-nlp--word-embeddings)
     - [Step-by-Step Worked Example: Text to Vectors](#step-by-step-worked-example-text-to-vectors)
     - [Skip-Gram vs. CBOW vs. GloVe](#skip-gram-vs-cbow-vs-glove)
   - [Real-World Applications (Sentinel, Speech, Video)](#17-real-world-applications-sentinel-speech-video)
3. [Quick Summary & Cheat Sheet](#18-quick-summary--cheat-sheet)

---

# PART A: Convolutional Neural Networks (CNN)

---

## 1. What is Convolution? (The Sliding Window Analogy)

Standard multi-layer perceptrons treat images like flat vectors of numbers. If you feed a $1000 \times 1000$ pixel RGB photo into a regular network, you get $3,000,000$ inputs! Connecting that to 1,000 hidden nodes requires **3 billion weights**—your computer would instantly run out of memory!

**CNNs solve this with Convolution.**

### Analogy: The Flashlight Inspection

Imagine looking at a huge wall poster in the dark. Instead of trying to see the entire poster at once, you take a **small flashlight** and shine it on a $3 \times 3$ patch in the top-left corner. You check for edges, then slide the flashlight right by 1 pixel, check again, and repeat until you reach the bottom-right corner.

```
INPUT IMAGE (5x5 Grid)           KERNEL / FILTER (3x3)         FEATURE MAP (3x3 Output)
+---+---+---+---+---+             +---+---+---+                +---+---+---+
| 1 | 1 | 1 | 0 | 0 |             | 1 | 0 | 1 |                | 4 | 3 | 4 |
+---+---+---+---+---+             +---+---+---+                +---+---+---+
| 0 | 1 | 1 | 1 | 0 |      *      | 0 | 1 | 0 |       =        | 2 | 4 | 3 |
+---+---+---+---+---+             +---+---+---+                +---+---+---+
| 0 | 0 | 1 | 1 | 1 |             | 1 | 0 | 1 |                | 2 | 3 | 4 |
+---+---+---+---+---+             +---+---+---+                +---+---+---+
| 0 | 0 | 1 | 1 | 0 |
+---+---+---+---+---+
| 0 | 1 | 1 | 0 | 0 |
+---+---+---+---+---+
```

---

### Convolution Output Dimension Formula & Examples

When sliding a kernel over an image, how big will the output feature map be?

$$\text{Output Width / Height} = \left\lfloor \frac{N - F + 2P}{S} \right\rfloor + 1$$

Where:
- $N$: Input image spatial size ($N \times N$)
- $F$: Kernel / Filter spatial size ($F \times F$)
- $P$: **Padding** (Number of zero-pixel borders added around the image)
- $S$: **Stride** (How many pixels the filter shifts per step)

#### Worked Example:
- Input Image: $32 \times 32$ ($N = 32$)
- Filter Size: $5 \times 5$ ($F = 5$)
- Padding: $2$ ($P = 2$)
- Stride: $1$ ($S = 1$)

$$\text{Output Size} = \frac{32 - 5 + 2(2)}{1} + 1 = \frac{32 - 5 + 4}{1} + 1 = 31 + 1 = \mathbf{32 \times 32}$$

> **Note on Padding:** When output size matches input size ($32 \rightarrow 32$), it is called **"Same Padding"**. If no padding is added ($P=0$), it is called **"Valid Padding"** (output shrinks).

---

### Common 3x3 Kernels

Kernels are small weight matrices that detect specific visual patterns:

```
    Sharpen Kernel               Edge Detection Kernel             Gaussian Blur Kernel
  +----+----+----+                 +----+----+----+                  +----+----+----+
  |  0 | -1 |  0 |                 | -1 | -1 | -1 |                  | 1  | 2  | 1  |  1
  +----+----+----+                 +----+----+----+                  +----+----+----+ ---
  | -1 |  5 | -1 |                 | -1 |  8 | -1 |                  | 2  | 4  | 2  | 16
  +----+----+----+                 +----+----+----+                  +----+----+----+
  |  0 | -1 |  0 |                 | -1 | -1 | -1 |                  | 1  | 2  | 1  |
  +----+----+----+                 +----+----+----+                  +----+----+----+
```

---

## 2. Core CNN Concepts: Sparse Connectivity, Weight Sharing, & Pooling

### A. Sparse Connectivity
Instead of connecting every single neuron to all 1,000,000 pixels, a CNN neuron only connects to a tiny local patch (e.g., $3 \times 3 = 9$ pixels). This drastically reduces the number of connections.

### B. Weight Sharing
Instead of creating brand new weights for every spot on the image, **the exact same $3 \times 3$ kernel matrix slides across the ENTIRE image**. 
- If a filter learns to detect a vertical edge in the top-left corner, that same filter will detect a vertical edge in the bottom-right corner!
- This gives CNNs **Translation Invariance** (a cat is recognized as a cat whether it appears in the top-left or bottom-right of a photo).

### C. Pooling (Downsampling)
Pooling reduces the height and width of feature maps to cut down memory usage and make features robust to small movements.

```
MAX POOLING (2x2 Filter, Stride 2)           AVERAGE POOLING (2x2 Filter, Stride 2)
  +----+----+----+----+                        +----+----+----+----+
  | 1  | 3  | 2  | 4  |                        | 1  | 3  | 2  | 4  |
  +----+----+----+----+   Max Select           +----+----+----+----+   Average
  | 5  | 6  | 1  | 2  |  ------------> [ 6  4 ] | 5  | 6  | 1  | 2  |  ---------> [ 3.75  2.25 ]
  +----+----+----+----+                   [ 8  3 ] +----+----+----+----+                 [ 3.00  1.00 ]
  | 7  | 8  | 0  | 1  |                        | 7  | 8  | 0  | 1  |
  +----+----+----+----+                        +----+----+----+----+
  | 2  | 1  | 3  | 0  |                        | 2  | 1  | 3  | 0  |
  +----+----+----+----+                        +----+----+----+----+
```

---

## 3. CNN Architecture & Hierarchy of Features

A full CNN pipeline stacks multiple stages:

```
INPUT IMAGE ===> [ CONV + RELU ] ===> [ POOL ] ===> [ CONV + RELU ] ===> [ POOL ] ===> [ FLATTEN ] ===> [ DENSE ] ===> OUTPUT
```

### Receptive Field & Feature Hierarchy
As you go deeper into the network, neurons "see" larger areas of the original image:

```
   Deep Layers (Layer 5+)     ====> Sees full objects (Dogs, Cars, Faces)
            ^
   Middle Layers (Layer 3-4)  ====> Sees object parts (Wheels, Eyes, Door handles)
            ^
   Early Layers (Layer 1-2)   ====> Sees basic features (Edges, Lines, Color gradients)
```

---

## 4. Evolution of CNN Architectures (LeNet to ResNet)

```
1998: LeNet-5     --> 5 layers, designed for zip code digits (MNIST).
2012: AlexNet     --> 8 layers, introduced ReLU, Dropout, and GPU training (won ImageNet).
2014: VGGNet      --> 16-19 layers, proved that stacking tiny 3x3 filters works best.
2014: GoogLeNet   --> 22 layers, introduced 1x1 convolutions & Inception parallel modules.
2015: ResNet      --> 50-152 layers! Solved vanishing gradients with Residual Skip Connections.
```

### The ResNet Skip Connection Breakthrough
In networks with >20 layers, accuracy degraded because gradients vanished. ResNet added a **Skip Connection**:

```
                  +--------------------------------+ (Identity Shortcut: + x)
                  |                                |
                  v                                v
Input (x) ---> [ Conv Layer ] ---> [ ReLU ] ---> [ Conv Layer ] ---> [ (+) Add ] ---> Output F(x) + x
```
Instead of forcing the network to learn an entire complex mapping $H(x)$, it only has to learn the **residual update $F(x) = H(x) - x$**. If a layer isn't needed, its weights drop to zero and the identity shortcut passes the data along cleanly!

---

## 5. Transfer Learning Strategies

Don't train a CNN from scratch! Take a model (like ResNet-50) already trained on 1.4 million ImageNet photos, and repurpose it for your custom problem (e.g., detecting plant diseases).

```
+-------------------------------------------------------------+-----------------------+
|    PRETRAINED CONVOLUTIONAL BASE (FROZEN WEIGHTS)           |  NEW CUSTOM HEAD      |
|    Learns generic edges, textures, shapes from ImageNet     |  Trainable Dense Layer|
+-------------------------------------------------------------+-----------------------+
|  Conv1 -> Conv2 -> Conv3 -> Conv4 -> Conv5                  |  FC -> Softmax Output |
+-------------------------------------------------------------+-----------------------+
```

### Strategy Selection Table:

| Dataset Size | Similarity to Original ImageNet Data | Recommended Strategy |
|:---:|:---:|:---|
| **Small** | High (e.g., Cats vs Dogs) | **Feature Extraction:** Freeze all conv layers; train only new output classification layer. |
| **Small** | Low (e.g., X-ray scans) | **Partial Fine-Tuning:** Freeze early conv layers; unfreeze later conv layers + output layer. |
| **Large** | High or Low | **Full Fine-Tuning:** Initialize with pretrained weights, train full network with tiny learning rate. |

---

## 6. Object Detection & Localization (R-CNN vs. YOLO)

- **Classification:** "Is there a car in this image?" (Outputs single label).
- **Localization:** "Where is the car?" (Outputs single bounding box $[x, y, w, h]$).
- **Detection:** "Find ALL cars, pedestrians, and bikes in this image!" (Outputs multiple boxes + labels).

```
R-CNN Family (Two-Stage Detectors)                 YOLO Family (One-Stage Detectors)
1. Propose ~2000 region proposals.                1. Divide image into S x S grid.
2. Run CNN on EACH region individually.           2. Single pass predicts boxes + classes instantly.
--> Very accurate, but SLOW!                      --> Super FAST! (Real-time 60+ FPS).
```

---

## 7. Face Recognition & Triplet Loss

To recognize faces, a network outputs a 128-dimensional numeric embedding vector $f(x)$.

### Triplet Loss Formula
During training, we present the network with 3 images at once:
- **Anchor ($A$):** Photo of Person X
- **Positive ($P$):** Another photo of Person X
- **Negative ($N$):** Photo of Person Y (Different person)

$$\mathcal{L}(A, P, N) = \max\left( 0, \|f(A) - f(P)\|^2 - \|f(A) - f(N)\|^2 + \alpha \right)$$

Where $\alpha$ is a safety margin enforcing that $A$ and $P$ are pushed close together, while $A$ and $N$ are pushed far apart.

---

## 8. Neural Style Transfer

Neural Style Transfer mixes the **Content** of one photo with the **Artistic Style** of another painting using a pretrained VGG-19 network.

```
Content Image (e.g., Photo of Eiffel Tower) ---\
                                                 +---> Minimizes Loss L_total = (a * L_content) + (b * L_style)
Style Image (e.g., Van Gogh Starry Night)   ---/
```

- **Content Loss ($L_{\text{content}}$):** Calculated from deep layer activations.
- **Style Loss ($L_{\text{style}}$):** Calculated using **Gram Matrices** (which capture feature channel correlations/textures across multiple layers).
- *Instead of updating weights, gradient descent updates the pixels of the synthesized image directly!*

---

# PART B: Recurrent Neural Networks (RNN)

---

## 9. Introduction to Sequential Data

Standard feedforward networks assume all inputs are independent. But for **sequential data**, order matters completely!
- Text: *"Not bad, actually good!"* vs. *"Good, actually bad!"*
- Time-series stock prices, audio speech, video frames.

---

## 10. RNN Architecture & The Memory Hidden State

An RNN contains a **feedback loop** that passes a hidden memory vector $h_t$ from step to step across time.

```
UNROLLED RNN IN TIME:
      y1                  y2                  y3
      ^                   ^                   ^
      |                   |                   |
   +-----+     W_hh    +-----+     W_hh    +-----+
   | h1  | ----------> | h2  | ----------> | h3  |  (Hidden Memory Vector)
   +-----+             +-----+             +-----+
      ^                   ^                   ^
      | W_xh              | W_xh              | W_xh
      x1                  x2                  x3
   (Time 1)            (Time 2)            (Time 3)
```

### Core Hidden State Equation:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

---

## 11. Backpropagation Through Time (BPTT) & Gradient Problems

To train an RNN, we unroll the sequence through all time steps and calculate gradients backward.

### The Vanishing / Exploding Gradient Problem
Because the exact same weight matrix $W_{hh}$ is multiplied repeatedly at every time step:
- If weights are $< 1.0$ $\rightarrow$ Gradients shrink to $0.0000$ after 10-20 steps (**Vanishing Gradient**). The network forgets early words!
- If weights are $> 1.0$ $\rightarrow$ Gradients blow up to infinity (**Exploding Gradient**). *Fix: Gradient Clipping (cap maximum gradient).*

---

## 12. Gated Recurrent Units (GRU)

GRUs fix vanishing gradients using **2 smart gates**:

```
      +----------------------------------------------------+
      |                                                    v
h_{t-1} ---> [ Reset Gate (r_t) ] ---> Candidate (\tilde{h}_t) ---> [ Update Gate (z_t) ] ---> h_t
```

1. **Update Gate ($z_t$):** Controls how much past memory $h_{t-1}$ to keep vs. overwrite.
2. **Reset Gate ($r_t$):** Controls how much past memory to forget when calculating new candidate information.

---

## 13. Long Short-Term Memory Networks (LSTM)

LSTMs solve vanishing gradients using a dedicated **Cell State ($c_t$)** (a high-speed memory highway running across time with zero multiplication obstruction).

```
                      Cell State Highway (c_t)
   c_{t-1} -----------------(x)--------------------(+)------------------> c_t
                             ^                      ^
                             | (Forget Gate f_t)    | (Input Gate i_t * \tilde{c}_t)
                             |                      |
   h_{t-1} ----+------> [ FORGET ]            [ INPUT ]           [ OUTPUT ] ----> h_t
               |
    x_t -------+
```

### The 3 LSTM Gates:
1. **Forget Gate ($f_t = \sigma(\dots)$):** Decides what information to drop from memory highway.
2. **Input Gate ($i_t = \sigma(\dots)$):** Decides what new information to add into memory highway.
3. **Output Gate ($o_t = \sigma(\dots)$):** Decides what parts of memory highway to output as hidden state $h_t$.

$$\text{Cell Update (Additive!): } c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

---

## 14. Encoder-Decoder (Seq2Seq) Models

Used for tasks where input length $\neq$ output length (e.g., Machine Translation).

```
ENCODER (English Input)                                DECODER (French Output)
"I" ---> "love" ---> "cats" ---> [ Context Vector (c) ] ---> "J'aime" ---> "les" ---> "chats"
```

---

## 15. Attention Mechanism & Self-Attention (Transformers Preview)

### The Context Bottleneck Problem
Squeezing a 100-word paragraph into one single fixed context vector $c$ causes the model to forget details.

### The Attention Solution
Instead of relying on a single context vector, **Attention allows the decoder to look at ALL encoder steps dynamically**, focusing only on the relevant input words for each output step!

```
Output Word "chats"  <== Calculates Weights (0.05, 0.05, 0.90) ==> Attends mostly to Input Word "cats"
```

### Self-Attention (Query, Key, Value) Formula:
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

---

## 16. Natural Language Processing (NLP) & Word Embeddings

Networks don't understand text strings. We convert words into dense numeric vectors called **Word Embeddings**.

### Step-by-Step Worked Example: Text to Vectors

Consider the sentence: `"the cat sat on the mat"`

#### Step 1: Tokenization & Vocabulary Building
Unique words $\rightarrow$ Vocabulary IDs:
$$\text{the} \rightarrow 0, \quad \text{cat} \rightarrow 1, \quad \text{sat} \rightarrow 2, \quad \text{on} \rightarrow 3, \quad \text{mat} \rightarrow 4$$

#### Step 2: One-Hot Vectors vs. Dense Embeddings
- One-Hot Vector for `"cat"` (ID = 1): $[0, 1, 0, 0, 0]^T$
- Dense Embedding Matrix $E$ ($5 \times 3$ dimensions):
  $$E = \begin{bmatrix} 
  0.2 & 0.8 & -0.1 \\ 
  0.9 & 0.3 & 0.5 \\ 
  -0.4 & 0.6 & 0.7 \\ 
  0.1 & -0.2 & 0.3 \\ 
  0.8 & 0.4 & 0.6 
  \end{bmatrix}$$
  - Embedding for `"cat"` = Row 1 = $[0.9, 0.3, 0.5]$.

---

### Skip-Gram vs. CBOW vs. GloVe

- **Skip-Gram (Word2Vec):** Takes center word $\rightarrow$ Predicts context words. (Great for small data / rare words).
- **CBOW (Word2Vec):** Takes context words $\rightarrow$ Predicts center word. (Faster to train).
- **GloVe:** Uses global word co-occurrence matrix factorization across the whole dataset.

> **Vector Arithmetic Power:** $\text{Vector("King")} - \text{Vector("Man")} + \text{Vector("Woman")} \approx \text{Vector("Queen")}$

---

## 17. Real-World Applications (Sentinel, Speech, Video)

1. **Sentinel Satellite Classification:** Combines CNN (to read spatial terrain features) + LSTM (to track seasonal crop growth over time).
2. **Speech Recognition:** Transforms raw audio waves $\rightarrow$ Mel Spectrograms $\rightarrow$ Deep RNN + CTC (Connectionist Temporal Classification) Loss $\rightarrow$ Text Transcript.
3. **Video Action Recognition:** 3D-CNNs (slide filters across height, width, AND video time frames) to detect actions like running, dancing, or jumping.

---

## 18. Quick Summary & Cheat Sheet

| Mechanism | Purpose / Key Formula | Main Advantage |
|:---|:---|:---|
| **Convolution** | $O = \lfloor \frac{N-F+2P}{S} \rfloor + 1$ | Extracts spatial patterns with far fewer parameters |
| **ResNet Skip** | $F(x) + x$ | Enables 100+ layer deep networks without vanishing gradients |
| **Transfer Learning** | Reuse pretrained weights (ImageNet) | High accuracy with tiny custom datasets |
| **YOLO** | Single-pass grid prediction | Real-time object detection (60+ FPS) |
| **LSTM** | Cell highway $c_t = f_t c_{t-1} + i_t \tilde{c}_t$ | Solves long-term text memory vanishing gradients |
| **Attention** | $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ | Focuses dynamically on relevant inputs; powers Transformers |
| **Word Embeddings** | Dense vector lookup matrix | Captures semantic relationships between words |