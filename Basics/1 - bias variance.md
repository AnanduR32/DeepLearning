# Bias and variance

- High variance, performing well on training but not on evaluation set - overfitting
- High bias - inability to capture relationship between dependents and target, performing bad on training as well as evaluation - underfitting

Epoch - A forward+backward pass on entire dataset

- Within an epoch there can be n iterations
- Training on each batch/subspace/sample of dataset.

Deeplearning takes approach of solving very complex problems with complex models and taking additional precautions to avoid overfitting.

## Ways to prevent overfitting:

- L1/L2 regularization - penalizing the weights
- Data management - Decrease complexity of model, reducing the number of layers, width of layer
- Data augmentation - To increase size of training data, eg: Images can be rotated, greyscaled, and so on.
- Dropout - At every iteration randomly selects n nodes and removes it along with it's incoming/outgoing connection
- Early stopping - A form of cross-validation, when performance isn't improving or is degrading - immediately stop training the model
- Weights decay - L2 normalization
- Batch normalization - Resetting the scale of the data at every layer, to prevent network from getting overwhelmed with very large incoming inputs from previous layer
  > $`\hat{x}_i=\frac{x_i-\mu_b}{\sqrt{\sigma_B^2+\epsilon}}`$  
  >
  > where,  
  > - 'Batch' Mean, $\mu_B$ -> $\frac{1}{m}\sum_{i=1}^m{x_i}$  
  > - 'Batch' Variance, $\sigma_B^2$ -> $\frac{1}{m}\sum_{i=1}^m(x_i-\mu_B)^2$  
  >  
  > Thus, new input $y_i$ = $\theta\hat{x}_i+\beta$  
  >
  > This allows for faster convergence and allows higher learning rates

### Classification of errors in ML model

- Reducible error  - Due to bias and variance error
  - bias: Simple model not explaining the data correctly (Underfitting)
  - variance: Model is too complex and explains training data well but doesn't generalize (Overfitting)
- Irreducible error - due to noise in data (Cannot be removed)

> Error = $`(\it{E}[\it{\hat{f}}(x)] - \it{f}(x))^2 + \it{E}[(\it{\hat{f}}(x) - \it{E}[\hat{\it{f}}(x)])^2] + \sigma_\epsilon^2`$  
> Error = $`Bias^2 + Variance + Irreducible\space Error`$  
> MSE = $`Bias^2 + Variance`$  
> E is "Expectation/Average"

### Example

House price predication

- high bias (Oversimplified assumption):

> The model doesn't understand the system/data correctly (i.e. doesn't understand the relationship between features -location, size, age, etc.), all predictions are wrong.  

Models having low bias algorithms: Decision Tree, k-Nearest Neighbours and SVM.  

- High variance (Overcomplicated assumption / Overly complex model):

> The model learns every detail of luxury villas but fails to generalize to the overall housing market.  

Models having high bias: Linear regression, Linear Discriminant Analysis and Logistic Regression.  

- Irreducible : 

> Someone pays extra due to sentimental value or undersells a house due to fear of it being haunted, this info cannot be accounted for in model.  

Ideal model - Low Bias and Low Variance.  
Achieve bias-variance tradeoff but adjusting for the cross-validation/training/mean error against the model complexity.

In DL/Neural networks to achieve the right sweetspot in bias-variance tradeoff, concentrate on:

- Dividing the data into train, test and validation splits
- Start with some network configuration with arbitrary depth and width of network, and adjust as per monitoring of the training and validation error, and generalization via test error.  
- Use activation functions in between the outputs of each layer  
  - Activation function: tanh, ReLU, leaky ReLU
  - Initialization method: He, Xavier
  - Optimization method: Adam  
- Randomized initialization of weights and bias of each node of neural network.

#### Decision matrix

| Training Error | Validation Error | Cause | Solution
| --- | --- | --- | ---
| High | High | High bias (Underfitting) | - Increase model complexity
| | | | - Train for more epochs
| Low | High | High Variance (Overfitting) | - Add more training data (eg: Data augmentation)
| | | | - Use regularization
| | | | - User early stopping (train less)
| low | low | Perfect tradeoff | - Ideal

