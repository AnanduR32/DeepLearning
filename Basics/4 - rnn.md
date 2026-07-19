# Sequential Learning

Study of machine learning algorithms designed for sequential data - time series, stock market, NLP, Speech transcription/translation, text generation, sentiment analysis, video activity recognition

Traditional feedforward network cannot comprehend this, as each input is assumed to be independent of each other, whereas in sequential data, each input is dependent on previous input.

- Elements in sequence can repeat
- Depends on contextual arrangement of data (ordering)
- Length of data varies (potentially infinitely)

eg:

- Text and sentences
- Audio, music generation
- Motion pictures, video activity recognition
- Timeseries data
- DNA sequence, protein structure
- material composition
- Decision making
- Sentiment classification

## RNNs

Primary distinguisher between CNNs and RNNs is ability to work with temporal or sequential data.

The nodes in hidden layer, instead of only feedforwarding the data, in RNN there is buildin feedback loop to loop information back to the same neuron multiple times, predefined number of times (it loops exactly once for every element (or time step t) in the input sequence), with a hidden state and input for specific contextual state (eg: timestamp) with activation function. These neurons are called recurrant units.

- The hidden states preserve information of long and varying sequences
- The hidden state captures the patterns/context of sequence into a summary vector
- The outputs are influenced by both input and hidden states

### DNN/CNN vs RNN
- | DNN/CNN | RNN
:--- | :--- | :---
**Objective** | Mapping the static input x to static output y | Model how probable a sequence is <br/> i.e. Model the probability of a sequence, or map an input sequence to an output sequence.
**Model** | $y{\approx}f_\theta(x)$ | $\Pi_{t=1}^Tp(x_t \mid x_{1}, \dots, x_{t-1}) \approx f_\theta(x_t, h_{t-1})$<br/>(where $h_{t-1}$ is the historical hidden state)<br/>Model using joint probabilities from the conditionals 
**Loss** | $L(\theta)=\sum_{i=1}^{N}l(f_{\theta}(x_i),y_i)$ | $L(\theta) = -\sum_{i=1}^{N} \sum_{t=1}^{T} \log p(x_{i,t} \mid x_{i,<t})$<br/>(Negative Log-Likelihood of the sequence)
**Optimization** | $\theta^* = \arg\min_\theta L(\theta)$ | $\theta^* = \arg\min_\theta L(\theta)$<br/>(Minimizing the negative log-likelihood maximizes probability)

RNNs incorporate context vectorizing to deal with large context windows - Converts input data in raw format to vectors (array of real numbers) - which is supported by ML models.

- This preserves order
- Allows handling varying inputs

### Modelling

- $x_t$ - input to network at timestamp t
- $h_t$ - Hidden state, representing a contextual vector at time t, which is calculated on basis of current input and previous hidden state (t-1 input) 

Thus, new state:
$$h_t=f_W(h_{t-1},x_t)$$

where,

- $f_W$ - Function with parameters 'W', eg: tanh

$$h_t=tanh(W_{hh}h_{t-1}+W_{xh}x_t)$$

#### Intuition

RNNs are basically a for-loop within another for-loop, Instead of having vector inputs as for regular DNNs, we use a matrix - such that each input to the Network is a vector instead of a scalar, and the hidden recurrent units process the batch of scalars (vector input).

1. Loop 1: The Batch Loop (Outer for-loop)
This loop iterates over your training samples. eg: Sentences or timeseries data which is fed into the network, this loop handles each sentence one by one

2. Loop 2: The Time-Step Loop (Inner for-loop)
This is the "recurrent" loop that handles sequential data. Instead of processing the whole sentence at once, this loop processes it one element at a time (e.g., word by word or time-step by time-step)

#### Requirements for RNN Model

> - Given two functions w.r.t $s_i$ and $y_i$
>   - $s_i = \sigma(U x_i + W s_{i-1} + b)$: <br/> Projection of input into hidden space using weight U and bias b, applying an activation function $\sigma$, and
>   - $y_i = O(V s_i + c)$: <br/> Maps the hidden state to final output $y_i$ using weight V and bias c, scaled by output function O.
> - The exact same weights (U, V, b, c) are cloned/shared across all time steps.

- Ensure output $y_t$ is dependent on previous input $y_{t-1}$
- Independent of context window, i.e. variable inputs
- For any input the weights are all shared, the function executed at each step is the same.

#### Backpropagration Through Time (BPTT)

| Property/Characteristic | Dimensions | Weights
| --- | --- | ---
| Input Vector ($x_i$) | $x_i \in \mathbb{R}^n$ (Feature or Vocab size) | $U \in \mathbb{R}^{d \times n}$ (Maps input $\rightarrow$ hidden space)
| Hidden State Vector ($s_i$) | $s_i \in \mathbb{R}^d$ (Hidden units capacity) | $W \in \mathbb{R}^{d \times d}$ (Maps history $\rightarrow$ hidden space)
| Output Prediction Vector ($y_i$) | $y_i \in \mathbb{R}^K$ ($K$ target classes) | $V \in \mathbb{R}^{K \times d}$ (Maps hidden $\rightarrow$ output space)
| Loss Evaluation | Categorical Cross-Entropy Loss | Biases: $b \in \mathbb{R}^d$, $c \in \mathbb{R}^K$

1. **Global loss**
$$L=\sum_{t=1}^TL_t$$
where, $L_t$ is the cross-entropy loss between predicted vector $y_t$ and true one-hot target vector.

2. Calculating gradients of **L** w.r.t
    - **V** <br/>
        Maps hidden space to output network, hence derivate only depends on the current step's loss
        $$\frac{\delta{L_t}}{\delta{V}}=\frac{\delta{L_t}}{\delta{y_t}}.\frac{\delta{y_t}}{\delta{V}}$$
        Since weights are shared across all steps, we can sum gradients across entire sequence:
        $$\frac{\delta{L}}{\delta{V}}=\sum_{t=1}^T\frac{\delta{L_t}}{\delta{y_t}}.\frac{\delta{y_t}}{\delta{V}}$$
    - **W** <br/>
        Hidden state $s_t$ depends on $s_{t-1}$, all the way back to start of sequence $s_0$<br/>
        To calculate derivate of loss **L** at step **t** w.r.t **W**, we need to take into account how the previous steps are influenced leading up to the current W.
        $$\frac{\delta{L}}{\delta{W}}=\sum_{k=1}^t\frac{\delta{L_t}}{\delta{s_t}}.\frac{\delta{s_t}}{\delta{s_k}}.\frac{\delta{s_k}}{\delta{W}}$$
        where, $\frac{\delta{s_t}}{\delta{s_k}}$ represents the unrolled sequential error propagation through hidden state chain from step 't' back to earlier step 'k'
        $$\frac{\delta{s_t}}{\delta{s_k}}=\frac{\delta{s_t}}{\delta{s_{t-1}}}.\frac{\delta{s_{t-1}}}{\delta{s_{t-2}}}...\frac{\delta{s_{k+1}}}{\delta{s_k}}$$
        i.e. if k = 0,
        $$\frac{\delta{s_t}}{\delta{s_k}}=\frac{\delta{s_t}}{\delta{s_{t-1}}}.\frac{\delta{s_{t-1}}}{\delta{s_{t-2}}}...\frac{\delta{s_{1}}}{\delta{s_0}}$$
        or,
        $$\frac{\delta{s_t}}{\delta{s_k}}=\Pi_{j=k+1}^t\frac{\delta{s_j}}{\delta{s_{j-1}}}$$
        Finally,
        $$\frac{\delta{L}}{\delta{W}}=\sum_{t=1}^T\sum_{k=1}^t\frac{\delta{L_t}}{\delta{s_t}}.(\Pi_{j=k+1}^t\frac{\delta{s_j}}{\delta{s_{j-1}}}).\frac{\delta{s_k}}{\delta{W}}$$
        and similarly for **U**,
        $$\frac{\delta{L}}{\delta{U}}=\sum_{t=1}^T\sum_{k=1}^t\frac{\delta{L_t}}{\delta{s_t}}.(\Pi_{j=k+1}^t\frac{\delta{s_j}}{\delta{s_{j-1}}}).\frac{\delta{s_k}}{\delta{U}}$$

### Issues in RNN

- Short-term memory <br/>
  Can't deal with longer sequences, at each timestep the old information is morphed by current step input, the same issue can be seen during backpropagation (BPTT) as well.
  > Information is constantly overwritten and diluted by current step hidden state calculation.
- Exploding gradients <br/>
  The calculation of loss and derivatives involves multiplication of weight **W** repeatedly leading to exponentially large values.
- Vanishing gradients <br/>
  If weights are low (<=1) then the same multiplication mentioned above leads to very low gradients

### LSTM (Long Short Term Memory Networks)

Concepts:

- Selectively Writting information
- Selectively Reading only necessary information
- Selectively Forgetting unnecessary information

These concepts are achieved by introducing Cell State $c_t$, which acts as dedicated long-term memory track running through the input sequence.

Gates acting as values are used to decide exactly how much information is allowed to be cleared from, addted to, or read out from the long term memory tracks.
> Gates are combination of linear layer and a sigmoid activation function $\sigma$, which outputs strictly between 0 and 1, such that the gate can act as a valve.

LSTM relies on 3 primary gates to manage memory workflow:

- Forget Gate: Removes information no longer useful <br/>
  Looks at current input $x_t$ and previous hidden state $h_{t-1}$ to decide what information from long-term history is no longer useful and should be deleted
  $$f_t=\sigma(W_f.[h_{t-1}, x_t] + b_f)$$
  > Intuition: Basically L1 regularization but in forward pass inorder to maintain the context purity  
  > Eg: Sentence about motorcycles, is original context, if different topic comes in that new topic will be ignored.
- Input Gate: Additional useful information to cell state is added <br/>
  Decides what new information should be stored into the long-term memory, this is done using **Gate Filtering** and **Candidate State** selection <br/>
  Gate Filtering<br/>
  > A sigmoid layer acts as a value deciding which parts of that new information candidate are actually worth saving:
  $$i_t=\sigma(W_i.[h_{t-1},x_t]+b_i)$$
  New Information candidate<br/>
  > A tanh layer creates a vector of completely new information extracted from the current step's input
  $$\tilde{c_t}=tanh(W_c.[h_{t-1}, x_t]+b_c)$$
- Output Gate: Additional useful information to cell state is added<br/>
  Generating the short-term hidden state ($h_t$) that be used to make actual character prediction and passed onto the next hidden layer.  
  > Determined by input and memory of the step (element-wise matmul)
  $$o_t=\sigma(W_o.[h_{t-1},x_t]+b_o)$$
  $$h_t=o_t\odot tanh(c_t)$$

#### Benefits of this architecture

- Fix vanishing gradient and exploding gradients  
Both of which are caused due to the multiplicative nature of the $\Pi$ function over the hidden state derivatives.  
But due to the presence of the cell state ($c_t$) which is purely additive in nature  

  > $c_t=f_t\odot c_{t-1}+i_t\odot \tilde{c_t}$

### Gated Recurrent Unit (GRU)

- Similar to LSTM, they use gates to control long term memory, but much simpler in design/architecture, in that it doesn't need a memory unit/call state to control information.
- Faster to train and requires less training data.
- Components/gates (2), Because they use fewer gates, they have fewer internal weight matrices to train. This makes them computationally lighter, faster to execute per epoch, and often less prone to overfitting on smaller datasets:
  - Update gate:  
    $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$
  - Reset gate:  
    $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$
- Current Memory content/Candidate Hidden State $\tilde{h_t}$  
  $$\tilde{h_t}=\tanh(W.[r_t\odot h_{t-1},x_t]+b)$$
- Final Hidden State Assembly $h_t$
  The update gate $z_t$ performs linear interpolation to blend past state and new candidate state together:
  $$h_t=(1-z_t)\odot h_{t-1}+z_t\odot \tilde{h_t}$$