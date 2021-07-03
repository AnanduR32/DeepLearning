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
The **Loss function** is given by <br> <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}\left(\hat{y}%2Cy\right) = %2D\left(ylog\hat{y} %2B \left(1 %2D y\right)log\left(1 %2D \hat{y}\right)\right)">  

**Intuition behind the loss function:**  
* For y = 1, we want <img src="https://render.githubusercontent.com/render/math?math=log\hat{y}"> to be as large as possible
* For y = 0, we want <img src="https://render.githubusercontent.com/render/math?math=log\left(1%2D\hat{y}\right)"> to be as large as possible   

Because in both cases since terms are enclosed in `-ve`, if the term is large enough, <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}"> is minimized.  

**Cost Function** <br> <img src="https://render.githubusercontent.com/render/math?math=\mathit{J}\left(w,b\right) = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}\left(\hat{y}^{(i)}%2Cy^{(i)}\right)">  
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
  For all nodes in the neural network in the hidden layers it is generally preferably to use tanh() activation function instead of sigmoid   
      <img src="https://render.githubusercontent.com/render/math?math=tanh(z)=\frac{e^{z}%2De^{%2Dz}}{e^{z}%2Be^{%2Dz}}">  
  With the exception of output layer in cases where <img src="https://render.githubusercontent.com/render/math?math=y\in\{0,1\}"> where we'd want <img src="https://render.githubusercontent.com/render/math?math=\hat{y}\in\{0,1\}">  
  
  But these functions, for very small values of slope scales slowly, due to which we instead use *rectified linear* function, wherein when the value of z is negative the derivative is 0, and positive otherwise.  
  
    
