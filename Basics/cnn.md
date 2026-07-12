# Convolution Neural Network

Designed to work on image datasets, They work by capturing spatial rationship between pixels in image.

- Core idea - Neighbouring datapoints close together in space (for images) or time (for time-series) are highly correlated. CNN finds local patterns that reoccurs in data.
- In terms of timeseries data: The immediately preceding datapoints conceptually have higher correlation, and a weighted average of these points is used.
  > But in practice: Kernel weights are learnable parameters. The model automatically learns which time delays are most important; it does not automatically favor the most recent data point.

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

## Images

- Represented using matrices
- Each color represented by 8bit value (Higher quantization equals better representation of image, better quality)
- Range (0-250)
- Images can be represented in
  - 1D array - Signals
  - 2D array - Pixels
  - 3D array - Volumn elements  
- The value assigned to specific datapoint in image data can be represented as a function of many variables including depth, color, time, etc.
- Higher resolution images (more datapoints) produces more detailed images
- Feature matching detecting and comparing features / points in an images
  - applications : tracking
- Linear transformation / Geometric transformation:
  - flipping', rotating, translation, cropping, zooming, greyscaling, modify contrast, brightness and so on, adding noise (blurring), mirroring.

### Color images

Representations -

| Total Bits | Reserves: Red | Reserves: Green | Reserves: Blue | Inference | Usage
| :--- | :--- | :--- | :--- | :--- | :---
| 8 | 3 | 3 | 2 | Highly compressed | in ultra-low-power edge AI applications
| 16 | 5 | 6 | 5 | High Color mode | Legacy Embeded Systems or High-Frame-Rate mobile tracking
| 24 | 8 | 8 | 8 | Standard True Color image processing | All modern computer vision projects

## Feature Extraction

### HOG (Histogram of Oriented Gradients)

- Used to extract features from images (Image descriptor)
- Detects gradient and orientation of edges
- Requires preprocessing
  - Aspect Ratio of image to be brought down to 1(width):2(height)
  - Ideal size of image to be 64 x 128 (base case as per paper published by Dalal & Triggs)
- This is done to facilitate the splitting of image into small connected regions/patches called cells
  - typically 8 x 8 pixels each, thus obtaining 8 x 16 grid
  - Cells can be grouped into larger, overlapping Blocks (usually 2 x 2 cells, or 16 x 16 pixels).
    - This block level is where Block Normalization happens to account for changes in illumination and shadowing
- Gradients are calculated:
  - In x and y direction separately
  - First at individual pixel level
  - Then patch level
  > $`G_x`$ = Change in X direction  
  > $`G_y`$ = Change in Y direction  
  > Direction: $`\tan{\theta}=\frac{G_y}{G_x}`$ i.e. `atan2(G_y, G_x)`  
  > Magnitude: $`\sqrt{G_x^2 + G_y^2}`$

### SIFT (Scale Invariant Feature Transform)

- A descriptor algorithm that is invariant to scaling, translation, rotation, partially to illumination changes as well.  
  - Tolerant to noise, change in viewing angle
  - A feature patch looks the same to the algorithm whether it's upside down, zoomed in, or slightly shadowed.
- Describes local features in image
- Can be said to be produced by gaussian convolution of image to identify key points in image
  - Progressively blurs it using Gaussian convolution at different scales
  - It also downsamples the image to create smaller versions (Octaves)
- Difference of Gaussian (DoG) - is what mathematically allows to identify the key points
  - Ideally should be done via Laplacean Laplacian of Gaussian (LoG) function, but it is computationally expensive
  - DoG simply finds difference between two adjacent scaled/blurred images in scale space, the resultant is approximation of LoG
  - The algorithm looks for "local extrema" or points that are the maximum or minimum compared to their 8 neighbors in the current  blur image, and their 9 neighbors in the scales above and below it - keypoints
- With each keypoints then the local magnitude and direction are calculated
  - creating a histogram whose highest peak becomes dominant orientation of the keypoint
- Creating descriptor 
  - The SIFT algorithm takes 16 x 16 neighbour around keypoints then further subdivides into 16 4 x 4 subblocks
  - For each subblock creates 8-bin orientation histogram
  - 16 subblocks x 8 bins = 128-dimensional vector -> The descriptor
  - This vector can then be normalized to handle illumination changes.

## Inherent properties of CNN

### Sparse connectivity

- In dense networks the image is flattened (1000x1000 resolution into a vector of size [1000 $\times$ 1000,1]) then each would would map to an individual node in the next layer and eventually reach output layer through set of hidden layers of user-defined widths.
- In CNN the image in 2D matrix is first reduced using the kernel and resultant of weighted sum is fed into the network (input layer)  
  At this point the node in layer maps to 9 nodes of 2D image (input) per kernel.  
  The number of nodes in next layer is reduced from [1000 $\times$ 1000,1] to [998 $\times$ 998,1] if kernel is of order 3x3 and stride 1  
  1 node in the next layer connects back to only 9 nodes in the input

### Weight Sharing

- In Dense Networks there is no weight sharing. Every single connection between an input node and a hidden node has its own unique, independent weight. If you have 1000 x 1000 input nodes and 500 hidden nodes, the network must learn 500 x 1000 x 1000 individual weight parameters.
- In CNNs: A single kernel (say 3 x 3 matrix containing just 9 weights) is duplicated and reused across the entire input image. As the kernel slides across the 1000 x 1000 input layer to calculate the values for the 998 x 998 nodes in the next layer, it uses the exact same 9 weights for every single patch.

### Pooling (Max pooling)

- Generally pooling layers are used to reduce size of inputs to speed up computation
- The kernal extracts the maximum/average value of the patch of image it is convolving
