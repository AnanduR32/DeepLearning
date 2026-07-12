# Hyperparameters

- Parameters/model parameters: configuration variables internal to models and model learns on it's own
  - Weights and coefficients of independent variables in any models, used by model for making predication. Learned by the model itself
- Hyperparameters: Explicitly defined by users to control the learning process
  - optimizable
    - Learning rate,
    - Number of epochs (Iterations per epoch is automatically calculated as Total Samples / Batch Size)
    - Activation functions
    - Mini-batch size
    - Regularization parameters
    - Loss Function
  - Model specific:
    - Depth and width of neural network

Hyperparameter tuning is mostly done on basis of the validation error

## Neural Net Hyper parameter Tunings

- Algorithms:
  - Vanilla/Momentum/Nesterov GD
  - AdaGrad
  - RMSProp
  - Adam
- Network Architecture
  - Number of layers (Depth)
  - Number of neuron (per layer) (Width)
- Activation Functions
  > - Non-linear activations are required to learn complex data representation.
  > - To reduce computational load, also optimize computation of gradients.
  > - Expected properties of activation function:
  >   - Non-linearity
  >   - Continuously differentiable
  >     - To be able to calculate rate of change of error (output) wrt weights (input), to calculate gradient during back-prop thus adjust weights
  >   - Range
  >     - eg: [0,1], [-1,1]
  >   - Monotonic
  >     - Output moves only in one direction relative to input i.e. either entirely non-decreasing or entirely non-increasing
  >     - Sign of derivative remains same and doesn't switch between iterations.
  >     - eg: ReLU, Sigmoid, Tanh. Non-monotonic: Swish, Mish
  >   - Non-vanishing gradient
  >     - The calculated gradients have significant impact on updates to coefficients. Always contributing the updates.
  >     - eg: In `sigmoid` activation When a neuron has reached either max or min val of range, we say function has saturated, and at saturated point the derivative would be near zero and doesn't contribute to updation - phenomenon is called vanishing gradient  Happens when weight becomes very large or very low thus mapping to output of activation function where outputs are saturated.
  >   - zero-centered function
  - Sigmoid
    - Vanishing gradient: When neuron's activation is 0 or 1, sigmoid neurons saturate i.e. Some neurons vanishes and no more learning happens (gradients at those ranges are near-zero or zero)
    - Non-Zero centered: Owing to it's range 0->1, the average value is ~0.5 this updates to gradients are always positive (+0.2, +0.15 ...) and zig-zag-y which is not optimal, and an average gradient update around zero is preferred (+0.3, -0.5, +0.2 ...)
  - tanh (RNN)
    - "Sigmoidal function"
    - range: [-1,1]
    - Zero-centered (as explained previously under drawback of sigmoid)
    - Suffers from vanishing gradient but not as much as sigmoid.
  - ReLU (CNN, DNN)
    - $`\it{f}(\it{x})=max(0,\it{x})`$
    - Range [0,x]
    - Not Zero-centered.
    - if x/input is < zero in forward pass, weights doesn't get updated in backward propagation
    - Faster to compute compared to Sigmoid/Tanh due to simple threshold.
    - Doesn't saturate in positive region
  - Leaky ReLU (CNN)
    - $`\it{f}(\it{x})=max(0.01\it{x},\it{x})`$
    - Range [0.0x,x] i.e (-inf, +inf)
    - Doesn't saturate in positive or negative region - large negatives are penalized to 0.01 times it's value.
    - It is closer to Zero Centered than ReLU, but not perfectly zero-centered.
    - Optimized performance as seen in regular ReLU
  - Softmax (in classification)
    - When needing to output as probability distribution, thus it is mostly used in the output layer.
    - range [0,1]
    - a.k.a. softargmax or multi class logistic regression.
    - $`\sigma(\vec{z})_i=\frac{e^{zi}}{\sum_{j=1}^K{e^{zj}}}`$
    - Width of output layer depends on number of classes, each neuron gives probability of input belonging that class `i`
    - This function to be only used with mutually exclusive classes
    - Algorithms in which softmax is used:
      - Neural networks
      - Multinomial logistic regression (softmax regression)
      - Bayes Naive classifier (NN - Single-Layer Perceptron with log-transformed inputs and a Softmax output layer)
      - Multiclass linear disciminant analysis
      - RNN to output probabilties of different actions to be taken
      - Attention mechanisms (Transformers)
- Strategies
  - Batching
  - Mini-Batch (32, 64, 128)
  - Stochastic
  - Learning rate schedule
- Initialization Methods
  - Xavier
  - He
- Regularization  
  > - Ability to generalize, preventing overfitting  
  > - Works on concept that smaller weights can lead to better models  
  > - Achieved by adding a penalization term to the cost function which limits the weights.
  - L2 (Ridge)  
    - Regularization term - $`\lambda = \Omega(\theta) = ||\theta||_2^2 = \theta_{11}^2 + \theta_{12}^2 + ...`$
    - Reduced to near zero but never zero, essentially decayingthe components of vector $\theta$ / w that do not contribute to reducing the cost function
  - L1 (Lasso)  
    - Regularization term - $`\Omega(\theta) = ||\theta||_1 = \sum_{i}{|\theta_i|}`$
    - Absolute value of weights are penalizied
    - Useful for image compression
    - Useful features have non-zero weights whereas others are kept zero.
    - Has property of built-in feature selection
    - In deep learning is used to shut off uninformative network connections.
  - Early stopping  
    - After certain number of epochs if validation error increasing though training error decreases.  
    - Training stopped when there is no longer any significant decrease in validation error compared to previous epoch
  - Dataset augmentation  
    > - Synthetic Data - generated artificiallly without using real world images, done using Generative Advarsarial Networks (GANs)
    > - Augmented Data - Derived from original real world images, achieved through Geometric transformation like 'flipping', 'rotating', 'translation', 'cropping', 'zooming', 'greyscaling', 'modify contrast, brightness and so on', 'adding noise' (blurring), thus increasing the diversity of dataset.
  - Dropout
    - It is a bagging method (averaging over several models to improve generalization)
    - In fully connected neural net, the model/network is prone to overfitting therefore randomly drop nodes/units during training
    - Based on a fixed dropout rate for each unit, independent of the rate for the other
    - dropout rate `p` ideally in range 20%-50%
    - This helps better generalize the model as each node adjust weights on it's own without being influence by weights of other neurons
  - Batch normalization

### Tuning learning rate

| Rate of decreasing of validation error | Learning rate
| -- | --
| Decreasing | Good
| Decreasing but jumping around | Volatile - Learning rate might be slightly too high; consider a learning rate scheduler or decay.
| Decreasing too slowly | Bad - Need to increase rate
| Increasing | Bad - Need to decrease rate
