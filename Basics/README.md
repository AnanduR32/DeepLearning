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

