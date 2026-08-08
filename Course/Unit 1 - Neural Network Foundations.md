# Unit I — Neural Network Foundations
> Course: 25CSA543A — Deep Learning for AI

---

## Single Layer Neural Networks (Perceptron)

A perceptron is the simplest neural network — one node that takes inputs, multiplies each by a weight, adds a bias, and passes through a step function.

$$y = \text{step}(w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b)$$

- If the weighted sum crosses a threshold → output 1, else output 0
- It's basically a linear classifier — draws a straight line (or hyperplane) to separate two classes

### Perceptron Learning Rule

The weight update is dead simple:
- If the perceptron predicts correctly → do nothing
- If it predicts 0 but should be 1 → increase the weights toward that input
- If it predicts 1 but should be 0 → decrease them

$$w_i \leftarrow w_i + \eta \cdot (y_{\text{true}} - y_{\text{pred}}) \cdot x_i$$

where $\eta$ is the learning rate. The perceptron convergence theorem guarantees this will find a solution *if the data is linearly separable*.

### The XOR Problem

This is the classic failure case. XOR outputs:

| $x_1$ | $x_2$ | XOR |
|-------|-------|-----|
| 0     | 0     | 0   |
| 0     | 1     | 1   |
| 1     | 0     | 1   |
| 1     | 1     | 0   |

No single straight line can separate the 1s from the 0s — try drawing it on paper. Minsky & Papert pointed this out in 1969 and it killed neural network research for over a decade. The fix? Add more layers.

---

## Multi-Layer Neural Networks (MLP)

An MLP stacks layers: input → one or more hidden layers → output. Each layer is fully connected to the next.

At each node in layer $l$:
- Compute $z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$ (linear combination)
- Apply activation: $a^{[l]} = g(z^{[l]})$ (non-linear transformation)

The input layer is $a^{[0]} = X$. The final layer's activation gives the prediction.

### Why depth matters

- A single hidden layer with enough neurons can theoretically approximate *any* continuous function (universal approximation theorem)
- But "enough neurons" can mean an absurd number. Deeper networks learn hierarchical features more efficiently — early layers learn edges, middle layers learn shapes, later layers learn objects
- In practice, 2–3 hidden layers handle most problems. Going deeper helps for complex tasks (images, language) but introduces training challenges (vanishing gradients, etc.)

### A concrete example

For XOR, a 2-layer network works:
- Hidden layer with 2 neurons: one learns AND, the other learns OR
- Output layer combines them: OR AND (NOT AND) = XOR
- Problem solved with just 2 hidden neurons — no single perceptron could do this

---

## Backpropagation

Backprop is just the chain rule applied systematically through the network. Nothing more, nothing less.

### Forward Pass

Push input through the network layer by layer:
1. $z^{[1]} = W^{[1]} X + b^{[1]}$, then $a^{[1]} = g(z^{[1]})$
2. $z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$, then $a^{[2]} = g(z^{[2]})$
3. Continue until the output layer, then compute loss $\mathcal{L}(a^{[L]}, y)$

### Backward Pass

Now propagate the error backwards to figure out how much each weight contributed to the loss.

For the output layer:
$$dz^{[L]} = a^{[L]} - y$$

For any layer $l$:
$$dW^{[l]} = \frac{1}{m} dz^{[l]} \cdot a^{[l-1]T}$$
$$db^{[l]} = \frac{1}{m} \sum dz^{[l]}$$
$$dz^{[l-1]} = W^{[l]T} \cdot dz^{[l]} * g'(z^{[l-1]})$$

The key intuition: each layer asks "how much did I contribute to the error?" and adjusts accordingly. The $g'(z)$ term is why activation function choice matters — if the derivative is tiny, gradients vanish as they flow back.

### Chain Rule Intuition

Think of it as a pipeline: if $f = h(g(x))$, then $\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$. Each layer multiplies its local gradient by the gradient flowing back from the layer above. That's literally all backprop is — automated chain rule.

### Worked example (single neuron)

Say $z = wx + b$, $a = \sigma(z)$, $\mathcal{L} = -(y \log a + (1-y) \log(1-a))$

- $\frac{d\mathcal{L}}{da} = \frac{a - y}{a(1-a)}$
- $\frac{da}{dz} = a(1-a)$ (sigmoid derivative)
- $\frac{d\mathcal{L}}{dz} = a - y$ (the two cancel nicely!)
- $\frac{d\mathcal{L}}{dw} = (a - y) \cdot x$

This is why sigmoid + binary cross-entropy is a natural pairing — the gradient simplifies cleanly.

### Worked example — full numeric backprop (2-layer network)

Let's do a **complete forward → loss → backward → update** pass with actual numbers. This is exactly what exams ask.

**Setup:** 1 input ($x = 0.5$), 1 hidden neuron, 1 output neuron, sigmoid everywhere. Target $y = 1$. Learning rate $\eta = 0.5$.

Initial weights and biases:

| Parameter | Value |
|-----------|-------|
| $w_1$ (input → hidden) | 0.3 |
| $b_1$ (hidden bias) | 0.1 |
| $w_2$ (hidden → output) | 0.7 |
| $b_2$ (output bias) | 0.2 |

**Step 1 — Forward pass:**

- Hidden layer: $z_1 = w_1 \cdot x + b_1 = 0.3 \times 0.5 + 0.1 = 0.25$
- Hidden activation: $a_1 = \sigma(0.25) = \frac{1}{1 + e^{-0.25}} = 0.5622$
- Output layer: $z_2 = w_2 \cdot a_1 + b_2 = 0.7 \times 0.5622 + 0.2 = 0.5935$
- Output activation (prediction): $\hat{y} = a_2 = \sigma(0.5935) = 0.6442$

**Step 2 — Compute loss** (binary cross-entropy):

$$\mathcal{L} = -(y \log \hat{y} + (1-y) \log(1-\hat{y})) = -(1 \times \log 0.6442 + 0) = 0.4402$$

**Step 3 — Backward pass (compute gradients):**

*Output layer:*
- $\delta_2 = a_2 - y = 0.6442 - 1 = -0.3558$
- $\frac{\partial \mathcal{L}}{\partial w_2} = \delta_2 \cdot a_1 = -0.3558 \times 0.5622 = -0.2000$
- $\frac{\partial \mathcal{L}}{\partial b_2} = \delta_2 = -0.3558$

*Hidden layer (chain rule through $w_2$):*
- $\delta_1 = (\delta_2 \cdot w_2) \times \sigma'(z_1) = (-0.3558 \times 0.7) \times (0.5622 \times 0.4378) = -0.2491 \times 0.2461 = -0.0613$
- $\frac{\partial \mathcal{L}}{\partial w_1} = \delta_1 \cdot x = -0.0613 \times 0.5 = -0.0307$
- $\frac{\partial \mathcal{L}}{\partial b_1} = \delta_1 = -0.0613$

**Step 4 — Update weights** ($w_{\text{new}} = w_{\text{old}} - \eta \cdot \text{gradient}$):

| Parameter | Old | Gradient | New = Old − 0.5 × Grad |
|-----------|-----|----------|------------------------|
| $w_2$ | 0.7 | −0.2000 | 0.7 − (0.5)(−0.2000) = **0.8000** |
| $b_2$ | 0.2 | −0.3558 | 0.2 − (0.5)(−0.3558) = **0.3779** |
| $w_1$ | 0.3 | −0.0307 | 0.3 − (0.5)(−0.0307) = **0.3153** |
| $b_1$ | 0.1 | −0.0613 | 0.1 − (0.5)(−0.0613) = **0.1307** |

**Step 5 — Verify improvement** (forward pass with new weights):
- $z_1 = 0.3153 \times 0.5 + 0.1307 = 0.2883$ → $a_1 = \sigma(0.2883) = 0.5716$
- $z_2 = 0.8000 \times 0.5716 + 0.3779 = 0.8352$ → $\hat{y} = \sigma(0.8352) = 0.6974$

Prediction moved from **0.6442 → 0.6974** (closer to target $y = 1$) ✅. Loss dropped from **0.4402 → 0.3604** ✅.

> **Exam tip:** The gradients are **negative** because the prediction was too low — the update *increases* all weights, pushing the output higher toward 1. If the prediction were too high ($\hat{y} > y$), gradients would be positive and weights would decrease.

### Worked example — multi-input backprop (matrix form)

Same idea, but now with 2 inputs and 2 hidden neurons — to show the matrix structure.

**Setup:** $x = \begin{bmatrix} 1 \\ 0.5 \end{bmatrix}$, target $y = 1$, sigmoid activation, $\eta = 0.1$.

$$W^{[1]} = \begin{bmatrix} 0.2 & 0.4 \\ 0.3 & 0.1 \end{bmatrix}, \quad b^{[1]} = \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix}, \quad W^{[2]} = \begin{bmatrix} 0.5 & 0.6 \end{bmatrix}, \quad b^{[2]} = \begin{bmatrix} 0.3 \end{bmatrix}$$

**Forward pass:**

$$z^{[1]} = W^{[1]} x + b^{[1]} = \begin{bmatrix} 0.2(1) + 0.4(0.5) \\ 0.3(1) + 0.1(0.5) \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.2 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.55 \end{bmatrix}$$

$$a^{[1]} = \sigma(z^{[1]}) = \begin{bmatrix} 0.6225 \\ 0.6341 \end{bmatrix}$$

$$z^{[2]} = W^{[2]} a^{[1]} + b^{[2]} = 0.5(0.6225) + 0.6(0.6341) + 0.3 = 0.9917$$

$$\hat{y} = a^{[2]} = \sigma(0.9917) = 0.7294$$

**Backward pass:**

$$\delta^{[2]} = a^{[2]} - y = 0.7294 - 1 = -0.2706$$

$$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \delta^{[2]} \cdot a^{[1]T} = -0.2706 \times \begin{bmatrix} 0.6225 & 0.6341 \end{bmatrix} = \begin{bmatrix} -0.1685 & -0.1716 \end{bmatrix}$$

$$\delta^{[1]} = (W^{[2]T} \cdot \delta^{[2]}) \odot \sigma'(z^{[1]}) = \begin{bmatrix} 0.5 \\ 0.6 \end{bmatrix} \times (-0.2706) \odot \begin{bmatrix} 0.2350 \\ 0.2320 \end{bmatrix} = \begin{bmatrix} -0.0318 \\ -0.0377 \end{bmatrix}$$

$$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \delta^{[1]} \cdot x^T = \begin{bmatrix} -0.0318 \\ -0.0377 \end{bmatrix} \begin{bmatrix} 1 & 0.5 \end{bmatrix} = \begin{bmatrix} -0.0318 & -0.0159 \\ -0.0377 & -0.0188 \end{bmatrix}$$

**Weight update** ($W_{\text{new}} = W_{\text{old}} - 0.1 \times \text{gradient}$):

$$W^{[1]}_{\text{new}} = \begin{bmatrix} 0.2 & 0.4 \\ 0.3 & 0.1 \end{bmatrix} - 0.1 \begin{bmatrix} -0.0318 & -0.0159 \\ -0.0377 & -0.0188 \end{bmatrix} = \begin{bmatrix} 0.2032 & 0.4016 \\ 0.3038 & 0.1019 \end{bmatrix}$$

$$W^{[2]}_{\text{new}} = \begin{bmatrix} 0.5 & 0.6 \end{bmatrix} - 0.1 \begin{bmatrix} -0.1685 & -0.1716 \end{bmatrix} = \begin{bmatrix} 0.5169 & 0.6172 \end{bmatrix}$$

> **Key pattern to notice:** the gradient magnitude shrinks as you go deeper ($\delta^{[1]}$ is ~10× smaller than $\delta^{[2]}$). This is the vanishing gradient effect in action — even with just 2 layers.

---

## Activation Functions

Without activation functions, stacking layers is pointless — you'd just get $W_2(W_1 x + b_1) + b_2 = W' x + b'$, a single linear transformation. Non-linearity is what gives depth its power.

### Comparison

| Function | Formula | Range | Pros | Cons |
|----------|---------|-------|------|------|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | (0, 1) | Smooth, probabilistic output | Vanishing gradient, not zero-centered |
| Tanh | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | (-1, 1) | Zero-centered | Still saturates at extremes |
| ReLU | $\max(0, z)$ | [0, ∞) | Fast, no saturation for $z>0$ | Dead neurons (if $z<0$ always) |
| Leaky ReLU | $\max(0.01z, z)$ | (-∞, ∞) | Fixes dead neuron problem | Small negative slope is arbitrary |

### Which to use?

- **Hidden layers**: ReLU is the default. It's fast and works well. Use Leaky ReLU if you're seeing dead neurons
- **Output layer**: depends on the task
  - Binary classification → sigmoid (outputs probability)
  - Multi-class → softmax
  - Regression → linear (no activation)
- **Tanh** is sometimes preferred over sigmoid in hidden layers because it's zero-centered, which helps gradient flow
- Sigmoid and tanh both suffer from vanishing gradients — for $|z| > 4$, the derivative is nearly zero, so learning stalls

### Why ReLU dominates

ReLU's gradient is either 0 or 1. No squishing, no saturation for positive values. This makes training much faster than sigmoid/tanh for deep networks. The "dead neuron" issue (neuron always outputs 0 if it gets stuck in the negative region) is real but usually manageable with proper initialization and learning rates.


---

## Gradient Descent

The core idea: compute the gradient of the loss w.r.t. parameters, then step in the opposite direction.

$$W \leftarrow W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}$$

The flavors differ in *how much data* you use per update and *how you adjust the step*.

### Batch vs Mini-batch vs SGD

| Variant | Batch size | Update frequency | Behavior |
|---------|-----------|-----------------|----------|
| Batch GD | Entire dataset ($m$) | Once per epoch | Smooth but slow, expensive for large data |
| Mini-batch GD | Typically 32–512 | $m / \text{batch\_size}$ times per epoch | Best of both worlds — fast and reasonably stable |
| SGD | 1 sample | $m$ times per epoch | Very noisy but can escape local minima |

- Mini-batch is what everyone actually uses. Batch sizes of 64, 128, 256 are common (powers of 2 for GPU efficiency)
- Pure SGD is too noisy in practice — you get a lot of oscillation
- Batch GD is too slow for large datasets — you wait forever before making a single update

### Momentum

Plain gradient descent oscillates a lot, especially in ravine-shaped loss surfaces (steep in one direction, shallow in another). Momentum smooths this out by accumulating a running average of past gradients.

On iteration $t$:
- $v_{dW} = \beta \cdot v_{dW} + (1 - \beta) \cdot dW$
- $W = W - \eta \cdot v_{dW}$

Think of it like a ball rolling downhill — it builds up speed in the consistent direction and dampens oscillation. $\beta = 0.9$ is the standard choice, which roughly averages over the last ~10 gradients.

### RMSProp

RMSProp adapts the learning rate per-parameter. Parameters with large gradients get smaller updates, and vice versa.

- $s_{dW} = \beta \cdot s_{dW} + (1 - \beta) \cdot dW^2$
- $W = W - \eta \cdot \frac{dW}{\sqrt{s_{dW}} + \epsilon}$

The $\epsilon$ (typically $10^{-8}$) prevents division by zero. This is great for dealing with sparse gradients and different parameter scales.

### Adam (Adaptive Moment Estimation)

Adam = Momentum + RMSProp. It's the go-to optimizer for most deep learning.

- First moment (mean): $v_{dW} = \beta_1 v_{dW} + (1-\beta_1) dW$
- Second moment (variance): $s_{dW} = \beta_2 s_{dW} + (1-\beta_2) dW^2$
- Bias correction: $v_{dW}^{\text{corrected}} = \frac{v_{dW}}{1-\beta_1^t}$, $s_{dW}^{\text{corrected}} = \frac{s_{dW}}{1-\beta_2^t}$
- Update: $W = W - \eta \cdot \frac{v_{dW}^{\text{corrected}}}{\sqrt{s_{dW}^{\text{corrected}}} + \epsilon}$

Standard hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. The bias correction matters early in training when the moment estimates are initialized to zero and would otherwise be too small.

### Learning Rate Decay

A fixed learning rate can overshoot the minimum during later stages of training. Decay strategies reduce $\eta$ over time:

- **Step decay**: halve $\eta$ every $k$ epochs
- **Exponential**: $\eta_t = \eta_0 \cdot e^{-\lambda t}$
- **1/t decay**: $\eta_t = \frac{\eta_0}{1 + \lambda \cdot t}$
- **Cosine annealing**: smoothly oscillates the learning rate — popular in modern training

Rule of thumb: start with a reasonable $\eta$ (e.g., 0.001 for Adam), train for a while, then add decay if the loss plateaus but doesn't converge.

---

## Train / Dev / Test Sets

### The split

- **Training set**: model learns from this
- **Dev (validation) set**: tune hyperparameters, pick the best model
- **Test set**: final unbiased evaluation — touch this *once*

Traditional split was 60/20/20, but with large datasets (>100k samples), you can get away with 98/1/1. The dev and test sets just need to be large enough to give statistically meaningful results.

### Distribution mismatch

This is a sneaky problem. If your training data comes from one distribution (e.g., high-quality web images) but your dev/test data comes from another (e.g., blurry phone photos), your model might do great on the dev set but fail in production.

Andrew Ng's advice:
- Dev and test sets **must** come from the same distribution — the distribution you actually care about
- Training data can be from a different distribution if that's all you have
- If there's a mismatch, create a "training-dev set" carved from training data. Compare:
  - High training error → underfitting (high bias)
  - Low training error, high training-dev error → overfitting (high variance)
  - Low training-dev error, high dev error → distribution mismatch

---

## Bias-Variance Trade-off

This is one of the most important diagnostic frameworks in ML.

- **High bias** = underfitting. The model is too simple to capture the patterns. Training error is high.
- **High variance** = overfitting. The model memorizes training data but doesn't generalize. Gap between training and dev error is large.

### Diagnosing the problem

| Training error | Dev error | Diagnosis |
|---------------|-----------|-----------|
| High | High | High bias (underfit) |
| Low | High | High variance (overfit) |
| High | Even higher | Both — high bias AND variance |
| Low | Low | Just right |

"High" and "low" are relative to the **Bayes error** (the best any model could do). If human-level performance on the task is ~1% error and your training error is 8%, you have a bias problem even though 8% sounds decent.

### Ng's Recipe

This is a practical flowchart for improving your model:

1. **High bias?** (training error is high)
   - Bigger network (more layers/neurons)
   - Train longer
   - Try a different architecture

2. **High variance?** (dev error >> training error)
   - More training data
   - Regularization (L2, dropout, data augmentation)
   - Try a different architecture

The key insight: in the deep learning era, these two problems can often be addressed independently. A bigger network almost always reduces bias without hurting variance (if you regularize). More data almost always reduces variance without hurting bias.

---

## Hyperparameter Settings

Deep learning has a lot of knobs. Knowing which ones matter most and how to search over them saves a ton of time.

### The important hyperparameters (roughly in order)

1. **Learning rate** ($\eta$) — the single most important hyperparameter. Too high → diverges. Too low → takes forever.
2. **Number of hidden units / layers** — controls model capacity
3. **Mini-batch size** — affects training speed and gradient noise
4. **Momentum** ($\beta$) — usually 0.9 works fine
5. **Learning rate decay** — matters more for fine-tuning
6. **Adam parameters** ($\beta_1, \beta_2, \epsilon$) — rarely need to change from defaults

### Search strategies

**Grid search** tries every combination of hyperparameters on a predefined grid. This is wasteful — if one hyperparameter matters much more than another (which is almost always the case), you waste most of your budget exploring irrelevant dimensions.

**Random search** samples combinations randomly from ranges you define. This is almost always better than grid because:
- You explore more unique values of each hyperparameter
- If learning rate matters but momentum doesn't, random search still gives you diverse learning rates

**Coarse-to-fine**: start with a wide random search, find the promising region, then zoom in with a tighter search around it.

### Logarithmic scale

Some hyperparameters should be searched on a log scale. Learning rate is the classic example — you want to explore 0.0001, 0.001, 0.01, 0.1, not 0.1, 0.2, 0.3, 0.4. Sample $r \in [-4, -1]$ uniformly, then set $\eta = 10^r$.

Similarly for $1 - \beta$: if $\beta \in [0.9, 0.999]$, search $1-\beta \in [0.001, 0.1]$ on a log scale, because the difference between 0.9 and 0.9005 is trivial, but between 0.999 and 0.9995 is significant.

### Weight Initialization

Bad initialization can kill training before it starts. If all weights are zero, every neuron computes the same thing — symmetry never breaks. If weights are too large, activations explode. Too small, they vanish.

**Xavier initialization** (good for sigmoid/tanh):
$$W^{[l]} \sim \mathcal{N}\left(0, \frac{1}{n^{[l-1]}}\right)$$

Scale weights by $1/\sqrt{n}$ where $n$ is the number of inputs to the layer. This keeps the variance of activations roughly constant across layers.

**He initialization** (good for ReLU):
$$W^{[l]} \sim \mathcal{N}\left(0, \frac{2}{n^{[l-1]}}\right)$$

The factor of 2 accounts for ReLU killing half the activations (the negative half). Without this correction, variance shrinks by half at each layer, and deep ReLU networks see vanishing activations.

Biases are typically initialized to zero — there's no symmetry issue with biases since the weights already break symmetry.

### Batch Normalization (quick note)

Batch norm normalizes the inputs to each layer (zero mean, unit variance) then applies learnable scale and shift parameters. This:
- Makes training more robust to initialization choices
- Acts as mild regularization
- Allows higher learning rates

It doesn't replace good initialization, but it makes the network much more forgiving of suboptimal choices.

---

## Quick Reference — Putting It All Together

When building a neural network from scratch, the workflow is:

1. **Architecture**: decide layers, units per layer, activation functions
2. **Initialize**: He init for ReLU layers, Xavier for sigmoid/tanh
3. **Forward pass**: compute predictions layer by layer
4. **Loss**: compute how wrong you are (cross-entropy for classification, MSE for regression)
5. **Backward pass**: backprop to get gradients
6. **Update**: apply optimizer (start with Adam)
7. **Evaluate**: check training vs dev error
8. **Diagnose**: use bias-variance framework to decide next steps
9. **Tune**: adjust hyperparameters (learning rate first, then architecture)
10. **Repeat** until dev error stops improving

The biggest mistakes beginners make:
- Spending too long tuning before checking for bugs (plot the loss curve first!)
- Not shuffling data before creating mini-batches
- Forgetting to normalize inputs (huge impact on convergence speed)
- Using sigmoid in hidden layers when ReLU would work better
- Not monitoring both training and dev loss — you can't diagnose without both