# AI
AI refers to the effort to automate intellectual tasks normally performed by humans, it makes use of set of rules and algorithms to figure out outcome based on those rules. Initally rules were hardwired manually using common logic, but this method of manually specifying the rules is time consuming and unscalable for various fields, thus to automate this process emerged the field of machine learning, and deep learning is just a form of machine learning wherein we use **neural networks** inspired on the structure and functioning of neurons in human body, data is represented in layers, "extracted through a multistage information extraction process".

Using the `Data and Answer we can derive the rules using machine learning techniques`, rather than the user giving the data and the rules to obtain the answer (which is classical programming techniques).  

The simplest form of a neural network is logistic regression for classification of objects.  

# Logistic Regression
Given,   
<img src="https://render.githubusercontent.com/render/math?math=\mathrm{W}\in\mathbb{R}^{n_{x}}">  
<img src="https://render.githubusercontent.com/render/math?math=\mathrm{X}\in\mathbb{R}^{n_{x}}">  
<img src="https://render.githubusercontent.com/render/math?math=\mathrm{b}\in\mathbb{R}">
  
If what we wish to predict is the probability of a class <img src="https://render.githubusercontent.com/render/math?math=\hat{y}"> given some features <img src="https://render.githubusercontent.com/render/math?math=x">, which can be represented as:  
<img src="https://render.githubusercontent.com/render/math?math=\hat{y} = \mathrm{P}\left(y|x\right)">  

Then we can have:  
<img src="https://render.githubusercontent.com/render/math?math=\hat{y} = \sigma\left(z\right)">  
Where, <img src="https://render.githubusercontent.com/render/math?math=z = w^{T}x%2Bb">
and,  
<img src="https://render.githubusercontent.com/render/math?math=\sigma\left(z\right) = \frac{1}{1%2Be^{z}}">  
  
### Cost function 
The **Loss function** is given by <br> 
  <p align="center">
    <br>
    <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}\left(\hat{y}%2Cy\right) = %2D\left(ylog\hat{y} %2B \left(1 %2D y\right)log\left(1 %2D \hat{y}\right)\right)">
    <br>
  </p>  

**Intuition behind the loss function:**  
* For y = 1, we want <img src="https://render.githubusercontent.com/render/math?math=log\hat{y}"> to be as large as possible
* For y = 0, we want <img src="https://render.githubusercontent.com/render/math?math=log\left(1%2D\hat{y}\right)"> to be as large as possible   

Because in both cases since terms are enclosed in `-ve`, if the term is large enough, <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}"> is minimized.  

**Cost Function** <br> 
  <p align="center">
    <br>
    <img src="https://render.githubusercontent.com/render/math?math=\mathit{J}\left(w,b\right) = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}\left(\hat{y}^{(i)}%2Cy^{(i)}\right)">
    <br>
  </p>  
For `m` training samples.  

The goal is to find the values of `w` and `b` that minimizes the cost function <img src="https://render.githubusercontent.com/render/math?math=\mathit{J}">, which can be accomplished using gradient descent.  

For the given cost function,<img src="https://render.githubusercontent.com/render/math?math=\mathit{J}">, which is a convex function, we can initialize the variables w and b, both to 0, and repeat the following steps which is a series of updation that finds the slope of the point on the function and adjusts the given parameters using this slope <img src="https://render.githubusercontent.com/render/math?math=\frac{d\mathit{J}(w,b)}{dw}"> with learning rate <img src="https://render.githubusercontent.com/render/math?math=\alpha"> for the parameter `w` and similarly <img src="https://render.githubusercontent.com/render/math?math=\frac{d\mathit{J}(w,b)}{db}"> for the parameter `b` till we reach the optimum minima.

**Algorithm**:  
```python
  while(True):
    w = w + learning_rate * slope_dJ_by_dw # Slope of cost function w.r.t w
    b = b + learning_rate * slope_dJ_by_db # Slope of cost function w.r.t b
```

**Intuition behind working of gradient descent**:   
Taking derivative of the cost function at a point (w,b) gives the surface(function) w.r.t these points, and the value of w and b are adjusted taking the derivative of J w.r.t w and b respectively.   
When we take derivative of a function say, <img src="https://render.githubusercontent.com/render/math?math=a^{2}"> we get <img src="https://render.githubusercontent.com/render/math?math=2a"> which tells us that for any value of a, if it were to be incremeneted by a very small value of the order of <img src="https://render.githubusercontent.com/render/math?math=10^{-\infinity}"> then the value of the resultant of the function increases by approximately <img src="https://render.githubusercontent.com/render/math?math=2a"> times. We apply this principle in figuring out the optimum minima for any given function by progressively decrementing/incrementing the values the parameters till we get `slope == 0`, which for a convex function is the optimum minima.  

Simple program for logistic regression:  
```python
m = 1000
J_array, b = np.zeros((m,1)), 0

np.random.seed(197)

w = np.zeros((1,2))
x_1 = np.random.randint(10, size = m).reshape(-1,m)
x_2 = np.random.randint(low = 25, high = 50, size = m).reshape(-1,m)
x = np.array([x_1,x_2]).reshape(2,m)

y = np.where(((x[1]<37.5) & (x[0]>5)), 1, 0)

for i in range(1000):
    z = np.zeros(m)
    a = np.zeros(m)

    z = np.dot(w,x) + b
    a = sigma(z)
    J = (-(y * np.log(a) + (1-y)* np.log(1-a))).mean()
    dz = a - y
    dw = (np.dot(x,dz.T).reshape(-1,2))/m
    db = dz.mean()
    w = w - alpha * dw
    b = b - alpha * db
    J_array[i] = J 

print(J_array[m-1])
```
o/p - [0.16131403]  

# Neural Network
Each layer comprises of one or more nodes and the `parameters` of node are represented with layer identity enclosed within `square brackets []`.   

At each layer of the neural network, that comprises of stack of nodes, comprise of both their own individual `z`, <img src="https://render.githubusercontent.com/render/math?math=z^{[l]}=W^{[l]}x%2Bb^{[l]}"> calculation as well as for `a`, <img src="https://render.githubusercontent.com/render/math?math=a^{[l]}=\sigma(z^{[l]})">  
The final output of the neural network is given by <img src="https://render.githubusercontent.com/render/math?math=a^{[l]}"> provided l is the final output layer of the network.  

Basics:
* <img src="https://render.githubusercontent.com/render/math?math=a^{[0]}"> represent the '*activations*' to the input layer of the neural network.  
* The parameters associated with each layer are <img src="https://render.githubusercontent.com/render/math?math=w^{[1]}"> and <img src="https://render.githubusercontent.com/render/math?math=b^{[l]}"> which are vectors of dimension k x j, were k is the number of nodes in that layer and j is the number of activation inputs, and column vector of dimension k x 1 respectively.  
* A two layer neural network comprises of the input layer (layer 0), hidden layer and output layer.  
* For input x, the parameters of the hidden layer and output layer can be represented in vectorized notation as:
    * <img src="https://render.githubusercontent.com/render/math?math=z^{[1]} = W^{[1]}a^{[0]}%2Bb^{[1]}">  
    * <img src="https://render.githubusercontent.com/render/math?math=a^{[1]} = \sigma(z^{[1]})">  
    * <img src="https://render.githubusercontent.com/render/math?math=z^{[2]} = W^{[2]}a^{[1]}%2Bb^{[2]}">  
    * <img src="https://render.githubusercontent.com/render/math?math=a^{[2]} = \sigma(z^{[2]})">,  
        
        Where,  
            <img src="https://render.githubusercontent.com/render/math?math=W^{[1]} = \left[w_{1}^{[1]T}x%2C w_{2}^{[1]T}x%2C w_{3}^{[1]T}x%2C w_{4}^{[1]T}x%2C \right]^{T}">, a 4<img src="https://render.githubusercontent.com/render/math?math=\times">3 matrix for 3 inputs (layer 0) and 4 nodes stacked in the hidden layer (layer 1),  
            and <img src="https://render.githubusercontent.com/render/math?math=a^{[0]}=x">  
    <img src="images/3Layer_NeuralNet.png">  
       
    Here, 
    <div display="block" overflow="auto">
      <table position="absolute" align="left">  
          <thead>
            <tr>
              <td>Parameter - Matrix</td>
              <td>Dimension</td>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><img src="https://render.githubusercontent.com/render/math?math=a^{0}"></td>
              <td>3 X 1</td>
            </tr>
            <tr>
              <td><img src="https://render.githubusercontent.com/render/math?math=w^{1}"></td>
              <td>4 X 3</td>
            </tr>
            <tr>
              <td><img src="https://render.githubusercontent.com/render/math?math=b^{1}"></td>
              <td>4 X 1</td>
            </tr>
            <tr>
              <td><img src="https://render.githubusercontent.com/render/math?math=a^{1}"></td>
              <td>4 X 1</td>
            </tr>
            <tr>
              <td><img src="https://render.githubusercontent.com/render/math?math=w^{2}"></td>
              <td>1 X 4</td>
            </tr>
            <tr>
              <td><img src="https://render.githubusercontent.com/render/math?math=b^{2}"></td>
              <td>1 X 1</td>
            </tr>
            <tr>
              <td><img src="https://render.githubusercontent.com/render/math?math=a^{2}"></td>
              <td>1 X 1</td>
            </tr>
          </tbody>
      </table>
      <div padding="2em" position="absolute" align="center" top="-100px">
        <img src="https://render.githubusercontent.com/render/math?math=a^{[1]}=\sigma\left(w^{[1]}%2Ea^{[0]}%2Bb^{[1]}\right)"><br>
        <img src="https://render.githubusercontent.com/render/math?math=(4,1)=\sigma\left((4,3)%2E(3,1)%2B(4,1)\right)"><br><br>
        <img src="https://render.githubusercontent.com/render/math?math=a^{[2]}=\sigma\left(w^{[2]}%2Ea^{[1]}%2Bb^{[2]}\right)"><br>
        <img src="https://render.githubusercontent.com/render/math?math=(1,1)=\sigma\left((1,4)%2E(4,1)%2B(1,1)\right)">
      </div>
    </div
  <br><br><br><br><br><br><br><br>   
  
  ### Activation functions  
  Activation functions is used to add a non-linear entity into the equation, whereas otherwise the resultant output of the neural network would simply be a linear combination of the inputs, and any additional hidden layer will not improve the model, and is quite redundant. We use linear(identity) function in case of regression models, eg: Predicting the house price given certain parameters, in this case the output layer activation function can be ReLU so that the output will always be <img src="https://render.githubusercontent.com/render/math?math=\gt0">    
  
  For all nodes in the neural network in the hidden layers it is generally preferably to use tanh() activation function instead of sigmoid   
      <img src="https://render.githubusercontent.com/render/math?math=tanh(z)=\frac{e^{z}%2De^{%2Dz}}{e^{z}%2Be^{%2Dz}}">  
  With the exception of output layer in cases where <img src="https://render.githubusercontent.com/render/math?math=y\in\{0,1\}"> where we'd want <img src="https://render.githubusercontent.com/render/math?math=\hat{y}\in\{0,1\}"> for binary classification  
  
  But these functions, for very small values of slope scales slowly, due to which we instead use `rectified linear unit` function, wherein when the value of z is negative the derivative is 0, and positive otherwise.  
  
  **Derivatives of common activation functions**:  
    * Sigmoid function: <img src="https://render.githubusercontent.com/render/math?math=g'(z) = \frac{d\left(g(z)\right)}{dz} = \frac{1}{1%2De^{%2Dz}}\left(1%2D\frac{1}{1%2De^{%2Dz}}\right) = a(1%2Da)">  
    * Tanh function: <img src="https://render.githubusercontent.com/render/math?math=g'(z) = \frac{d\left(g(z)\right)}{dz} = 1%2D\left(tanh(z)\right)^{2} = 1%2Da^{2}">  
    * ReLU:  
          <img src="https://render.githubusercontent.com/render/math?math=g'(z) = \left(\begin{align*}0 if z<0 \\1 if z>0\end{align*}\right)">
    * Leaky ReLU:  
          <img src="https://render.githubusercontent.com/render/math?math=g'(z) = \left(\begin{align*}0.01 if z<0 \\1 if z>0\end{align*}\right)">
  
  ### Equations: 
    * **Forward pass**:   
         * <img src="https://render.githubusercontent.com/render/math?math=Z^{[1]} = W^{[1]}%2EX%2BB^{[1]}">  
         * <img src="https://render.githubusercontent.com/render/math?math=A^{[1]} = tanh(Z^{[1]})">  
         * <img src="https://render.githubusercontent.com/render/math?math=Z^{[2]} = W^{[2]}%2EA^{[1]}%2BB^{[2]}">  
         * <img src="https://render.githubusercontent.com/render/math?math=A^{[2]} = sigmoid(Z^{[2]})">  
    * **Backward pass**:  
         * <img src="https://render.githubusercontent.com/render/math?math=dZ^{[2]} = A^{[2]} - Y">  
         * <img src="https://render.githubusercontent.com/render/math?math=dW^{[2]} = \frac{dZ^{[2]}%2EA^{[1]T}}{m}">  
         * <img src="https://render.githubusercontent.com/render/math?math=dB^{[2]} = \frac{\mathrm{np.sum}(dZ^{[2]}%2Ckeepdims=True%2Caxis=1)}{m}">  
         * <img src="https://render.githubusercontent.com/render/math?math=W^{[2]} = W^{[2]}%2D\mathit{alpha}*dW^{[2]}">  
         * <img src="https://render.githubusercontent.com/render/math?math=B^{[2]} = B^{[2]}%2D\mathit{alpha}*dB^{[2]}">  
         * <img src="https://render.githubusercontent.com/render/math?math=dZ^{[2]} = W^{[2]}%2EdZ^{[2]}*(1%2DA^{[1]^{2}})">  
         * <img src="https://render.githubusercontent.com/render/math?math=dW^{[1]} = \frac{dZ^{[1]}%2EX^{T}}{m}">  
         * <img src="https://render.githubusercontent.com/render/math?math=dB^{[1]} = \frac{\mathrm{np.sum}(dZ^{[1]}%2Ckeepdims=True%2Caxis=1)}{m}">  
         * <img src="https://render.githubusercontent.com/render/math?math=W^{1} = W^{[1]}%2D\mathit{alpha}*dW^{[1]}">  
         * <img src="https://render.githubusercontent.com/render/math?math=B^{1} = B^{[1]}%2D\mathit{alpha}*dB^{[1]}">  
  
  ### Backpropagation derivatives
    <img width=49% src="images/backprop_slope_derivation_1.jpg">
    <img width=49% src="images/backprop_slope_derivation_2.jpg">  

  ### Random initialization of weights
    if the weights are initialized to the same weight i.e. 0, then <img src="https://render.githubusercontent.com/render/math?math=a^{[1]}_1 = a^{[1]}_2"> for a given layer in the neural network, and consequently the values of <img src="https://render.githubusercontent.com/render/math?math=dz^{[1]}_1 = dz^{[1]}_2">  
    The weights should be initialized by random very small weights, furthermore the weights should be of the order of <img src="https://render.githubusercontent.com/render/math?math=10^{-2}"> such that the sigmoid function doesn't malfunction.
    
 
## Deep Neural networks 
  Each layer in the network tries to learn features whose complexity increases with each progressive (deeper) layer.  
  Eg: In case of image recognition, the first layer works at feature detection, identifying edges and their orientation, based on the number of nodes in it, the next layer identifies objects in image formed by grouping up of edges, which in case of facial recognition could be used to recognise features of face such as shapes of nose, ear, eyes etc.   
  And then finally the network can put together the different features and construct different types of faces with different structure and featues.   
  
  In case of audio/speech recognition, the first layer identifies the low level features of the audio, it's waveform, pitch, reverberations and so on, and the next layer puts together these detected features to construct words and then the next to construct sentences.  
  
  The earlier layers compute relatively simpler functions of the input whereas the deeper layers compute more complex functions.  
  
## Hyper parameters
  Whiles the **weights**, **bias terms** etc are the parameters to the neural network, hyperparameters are those variables that specify how the parameters evolve and to what extent they do during the training process,   
  Hyperparameters:   
  * learning rate <img src="https://render.githubusercontent.com/render/math?math=\left(\alpha\right)">   
  * Number of Iterations  
  * Number of hidden layers in the network <img src="https://render.githubusercontent.com/render/math?math=\left(\mathrm{L}\right)">  
  * Number of hidden units in each layer <img src="https://render.githubusercontent.com/render/math?math=\left(n^{[l]}\right)">  
  * Choice of activation functions (for each layer) <img src="https://render.githubusercontent.com/render/math?math=\left(g^{[l]}\right)">   
  * Momentum  
  * Min-batch size  
  * Regularization parameters <img src="https://render.githubusercontent.com/render/math?math=\left(\lambda\right)">  
  
  
# Regularization for neural networks   
  To reduce overfitting on training dataset, to come up with a model that has high accuracy, as well as generalises well on unseen data.  
  Types:  
  * L2 regularization (most widely used)    
  
      Penalizing on the cost function <img src="https://render.githubusercontent.com/render/math?math=\mathrm{J(m)}">, with additional term <img src="https://render.githubusercontent.com/render/math?math=\lambda">  
      As in,  
          <p align="center">
            <br>
              <img src="https://render.githubusercontent.com/render/math?math=\mathrm{J(...)} = \mathcal{L}\left(\hat{y}^{(i)}%2Cy^{(i)}\right)%2B\frac{\lambda}{2m}\sum_{l}\parallel w^{[l]}\parallel^{2}_{f}">
            <br>
          </p>  
  
      The penalizing term of L2 regularization in neural networks in known as *Frobenius norm* which simply the *Eucledian norm* but for matrices instead of vectors.  
  
      The penalizing term increases the cost function, reducing the weights further, <img src="https://render.githubusercontent.com/render/math?math=w \rightarrow 0">   
      Making the overall network more linear, unable to fit complex functions.  
  * Dropout regularization (Inverted dropout)
  
      Probabilities are assigned to each node in the network and they are removed randomly based on these probabilities from each iteration.  
  
      A common implementation of dropout regularization is the "*Inverted dropout*":  
        For a specific layer n <img src="https://render.githubusercontent.com/render/math?math=l^{n}">, create a dropout vector <img src="https://render.githubusercontent.com/render/math?math=d^{n}">, which is a boolean vector of same dimensions as the output of the layer <img src="https://render.githubusercontent.com/render/math?math=a^{n}">, with boolean values computed from randomized probabilities and relational operator to check if this value is above a certain threshold "keep_prob", which is the probability to decide whether that node is dropped out or not.  
      
      <p align="center">
        <br>
        <img src="https://render.githubusercontent.com/render/math?math=d^{n} = \mathrm{np.random.rand}(a^{n}.shape[0],a^{n}.shape[1])\lt \mathrm{keep_prob}">   
      </p>
      If keep_prob = p, then there's a (1-p) chance that the corresponding node will be removed from the layer based on the dropout vector values.    
  
      To simulate the dropout, we nullify the corresponding output value in vector <img src="https://render.githubusercontent.com/render/math?math=a^{n}"> by multipling the original <img src="https://render.githubusercontent.com/render/math?math=a^{n}"> with the dropout vector.  
      <p align="center">
      <img src="https://render.githubusercontent.com/render/math?math=\mathrm{np.multiply}(a^{n}%2Cd^{n})">  
      </p>
      Equivalent to: <img align="center" src="https://render.githubusercontent.com/render/math?math=a^{n}*=d^{n}">   
  
      Finally <img align="center" src="https://render.githubusercontent.com/render/math?math=a^{n}"> is multiplied by <img align="center" src="https://render.githubusercontent.com/render/math?math=\frac{1}{keep_prob}">, which is done so as to bump up the value of <img align="center" src="https://render.githubusercontent.com/render/math?math=z^{n%2D1}"> from the missing/shut-off nodes.  
      
      This technique is used by default in computer vision learning models.  
  
  * Data augmentation to increase the training data volume  
  * Early stopping after n iterations at the point at which dev-set error which had been decreasing till that point along with the train-set error, starts increasing instead. Which indicates that the model was performing optimally for the train-set as well as dev-set for the said number of iteration, but then overfit on the training data and it's performance will progressively decrease for the proceeding iterations for the dev-set.    
  
# Handling explosion/vanishing of gradients 
  
  Consider a very deep neural network with identity activation function and zero bias term *b*, then the overall output of the network is just <img align="center" src="https://render.githubusercontent.com/render/math?math=a^{l}=w^{l-1}*w^{l-2}*...*w^{3}*w^{2}*w^{1}*">   
  If the value of the gradients are slightly greater than identity matrix, then the gradients explode exponentially, and if less, then they vanish - diminishing in value exponentially.  
  
  To get around this problem, weight initialization for a layer *l* is done as, 
  <p align="center">
    <br>
    <img src="https://render.githubusercontent.com/render/math?math=w^{l} = \mathrm{np.random.rand}(shape)*\mathrm{np.sqrt}\left(\frac{1}{n^{l-1}}\right)">
    <br>
  </p>
  For relu activation function,
  <p align="center">
    <br>
    <img src="https://render.githubusercontent.com/render/math?math=w^{l} = \mathrm{np.random.rand}(shape)*\mathrm{np.sqrt}\left(\frac{2}{n^{l-1}}\right)">
    <br>
  </p>
  is found to perform better.  
  
# Gradient checking in neural network
  To check whether the implementation of the neural network is sound and that the gradients are being calculated correctly.
  
  Take all the weights and bias terms in the order, <img src="https://render.githubusercontent.com/render/math?math=w^{[1]}%2Cb^{[1]}%2Cw^{[2]}%2Cb^{[2]}%2E%2E%2E%2Cw^{[l]}%2Cb^{[l]}"> and reshape into a <img src="https://render.githubusercontent.com/render/math?math=\mathrm{m}\times1"> vector, and concatenate them into a large vector <img src="https://render.githubusercontent.com/render/math?math=\theta">.
  
  Similar create another large vector <img src="https://render.githubusercontent.com/render/math?math=\mathrm{d}\theta"> using the set <img src="https://render.githubusercontent.com/render/math?math=\mathrm{d}w^{[1]}%2C\mathrm{d}b^{[1]}%2C\mathrm{d}w^{[2]}%2C\mathrm{d}b^{[2]}%2E%2E%2E%2C\mathrm{d}w^{[l]}%2C\mathrm{d}b^{[l]}">  
  
  Then we check whether <img src="https://render.githubusercontent.com/render/math?math=\mathrm{d}\theta"> is the gradient or slope of the cost function <img src="https://render.githubusercontent.com/render/math?math=\mathrm{J}(\theta)">  
  
  Create a loop, which iterate `len`<img src="https://render.githubusercontent.com/render/math?math=(\theta)"> times, computing the values of vector <img src="https://render.githubusercontent.com/render/math?math=\mathrm{d}_{approx}^{[i]}">  
  
  <p align="center">
    <br>
    <img src="https://render.githubusercontent.com/render/math?math=\mathrm{d}_{approx}^{[i]} = \frac{\mathrm{J}(\theta_{1}, \theta_{2}, ... \theta_{i} %2B \epsilon, ...) - \mathrm{J}(\theta_{1}, \theta_{2}, ... \theta_{i} %2D \epsilon, ...)}{2\epsilon}">
    <br>
  </p>  
  
  Then, <img src="https://render.githubusercontent.com/render/math?math=\mathrm{d}_{approx}\theta">is compared with<img src="https://render.githubusercontent.com/render/math?math=\mathrm{d}\theta">to check their relative similarity using euclidean distance measure.    
  <p align="center">
    <br>
    <img src="https://render.githubusercontent.com/render/math?math=\frac{\parallel\mathrm{d}_{approx}\theta%2D\mathrm{d}\theta\parallel_{2}}{\parallel\mathrm{d}_{approx}\theta\parallel_{2}%2B\parallel\mathrm{d}\theta\parallel_{2}}">
    <br>
  </p>  
  
  If the resultant similarity is of the order of:  
  * Less than or equal to <img src="https://render.githubusercontent.com/render/math?math=10^{-7}">, it is ideal scenario and the gradient approximation is sound  
  * Between <img src="https://render.githubusercontent.com/render/math?math=10^{-7}"> and <img src="https://render.githubusercontent.com/render/math?math=10^{-3}">, it is moderately sound  
  * Greater than or equal to <img src="https://render.githubusercontent.com/render/math?math=10^{-3}">, implies unsound approximation.  
  
  Note:  
  * Gradient checking doesn't work when dropout regularization is implemented.  
  
  
    
# Mini-Batch gradient descent 
  
  The gradient descent parameter optimization for very large dataset (Bigdata) tends to perform very slow or unable to execute altogether due to resource requirements not being met, therefore to address this issue the mini-batch gradient descent is introduced wherein we have split the entire training dataset into multiple batches and perform gradient descent on each batch whilst updating the parameters in each mini-batchs' iteration (epoch). 
  
    Epoch - The number of times a certain set of training data is used to train the model, generally we use more epochs to improve the convergence of the model.  
  
  Types:  
    * When the batch size = m, then it denotes that the size of a single sub-batch is equal to the size of the training dataset, hence it acts as a typical batch gradient descent algorithm   
    * When the batch size = 1, the gradient descent is referred to as "Stochastic gradient descent" wherein each row in the training dataset/each example acts as a mini-batch on it's own.  
  
# Exponentially weighted (moving) averages
  A better performing optimization algorithm compared to mini-batch gradient descent.  
  
  Computed as,   
  <img src="https://render.githubusercontent.com/render/math?math=V_{t}=\beta V_{t-1}%2B(1-\beta)\theta_{t}">  
  
  where,   
  <img src="https://render.githubusercontent.com/render/math?math=\beta V_{t-1}"> represents the weighted average of the previous days for which the parameter <img src="https://render.githubusercontent.com/render/math?math=\beta"> decides the significance of, in the present/current weighted average, a higher values entails weighted average to be taken on a larger set of training samples, whereas a lower values produces more finely tuned weighted average across the dataset. 
  
  The larger the value of <img src="https://render.githubusercontent.com/render/math?math=\beta"> is the larger the time duration over which the reponse is averaged for.  
  
  eg: For a model that fits the temperature over an year, when <img src="https://render.githubusercontent.com/render/math?math=\beta = 0.9"> then the weight of the temperatures drops exponentially by a factor of <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{3}^{rd}"> for every 10 days. And for <img src="https://render.githubusercontent.com/render/math?math=\beta = 0.98">, that duration increares to 50 days (or which ever unit time duration that is used for measurement)  
  This increase is owing to the fact that a lot of the weight now is being given to the previous values that chains recursively for a larger period.  
  A larger value for <img src="https://render.githubusercontent.com/render/math?math=\beta">, say 0.5, makes the model susceptible a lot of noise degrading the model performance, but such a model adapts much more quickly to the response (prediction).  
  Based on the generalized equation <img src="https://render.githubusercontent.com/render/math?math=(1%2D\varepsilon)^\left(\frac{1}{\varepsilon}\right) = \frac{1}{e}">
  
  ### Implementing exponentially weighted averages 
  for day 0, <img src="https://render.githubusercontent.com/render/math?math=\mathrm{V}_{\theta} = 0"> and then progressively update the value of <img src="https://render.githubusercontent.com/render/math?math=\mathrm{V}_{\theta}">, for the corresponding daily temperature <img src="https://render.githubusercontent.com/render/math?math=\theta"> using the equation: <img src="https://render.githubusercontent.com/render/math?math=\mathrm{V}_{\theta} = \beta\mathrm{V}_{\theta_{prev}}%2B(1%2D\beta)\theta_{i}">   
  This implementation is memory efficient since it doesn't need to store all the values for <img src="https://render.githubusercontent.com/render/math?math=\mathrm{V}_{\theta}"> but continuously updates it's value, but is inefficient performance-wise for which we must compute average for a larger set of data.  
  
  ### Bias correction
  Bias issue occurs due to the original value of <img src="https://render.githubusercontent.com/render/math?math=\mathrm{V}_{\theta}"> being 0. 
  
  
  
  
  
  
          
    
