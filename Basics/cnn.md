# Convolution Neural Network

Designed to work on image datasets, They work by capturing spatial rationship between pixels in image.

- Core idea - Neighbouring datapoints close together in space (for images) or time (for time-series) are highly correlated. CNN finds local patterns that reoccurs in data.
- In terms of timeseries data: The immediately preceding datapoints conceptually have higher correlation, and a weighted average of these points is used.
  > But in practice: Kernel weights are learnable parameters. The model automatically learns which time delays are most important; it does not automatically favor the most recent data point.

## Images

- Represented using matrices
- Each color represented by 8bit value
- Range (0-250)

## Convolution Operator

> Convolution between two functions in mathematics produces a third function expressing how the shape of one function is modified by other  

- A specialized linear operator, it is a small square weight matrix  (3x3, or 5x5) or cube (timestamped series images/video) weight matrix  (N x N x T - t is duration of series) that scans the input data in 'strides' from top left to bottom right, to facilitate this, 'padding' is added to the input image
- This linear operator is a.k.a 'Kernel' or 'Filter'
- Process:
  - when scanning the kernel overlaps a patch of input
  - The corresponding pixel value is multiplied against the weight in the kernel
  - The multiplicative results are then summed up, often along with a bias term, to form a singular scalar sum
  - The sum then goes into a feature map.
- Output dimension:
  - input size - N x N
  - kernel size - F x F
  - Padding - P
  - Stride - S
  - Output image dimension: $`\frac{N-F+2P}{S}+1`$
- In terms of timeseries data: The weights are lower as we go deeper into the kernel (backward in time), the foremost slice of kernel will have higher weightage

> For a 1D dataset with 1D kernel,  
>
> $`g(x)=f(x)*h(x)`$  
> $`\int_{-\inf}^{\inf}f(s)*h(x-s)\delta{s}`$  
>
> Where,  
> h(x) is function that slides the filter across the input  
> f(x) is convolution of filter with overlapping input datapoints  
> s is dummy variable  
> g(x) shows how similar the input is with the kernel, as the final output has more resemblance to the kernel used

### Convolution Kernals

1. Original  
$`\begin{bmatrix}
0 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 0
\end{bmatrix}`$
2. Gaussian Blur - Minimizing the difference between central pixel and surrounding ones.  
$`\frac{1}{n}\begin{bmatrix}
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1 \end{bmatrix}`$
(or)
$`\frac{1}{n}\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1 \end{bmatrix}`$  
3. Sharpen - Amplifying the difference between central pixel and surrounding ones.  
$`\begin{bmatrix}
0  & -1 &  0 \\
-1 & 5  & -1 \\
0  & -1 &  0
\end{bmatrix}`$  
4. Edge detection  
$`\begin{bmatrix}
-1 & -1 & -1 \\
-1 &  8 & -1 \\
-1 & -1 & -1
\end{bmatrix}`$
